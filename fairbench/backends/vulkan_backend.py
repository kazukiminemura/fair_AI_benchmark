"""AMD INT8 GEMM using Vulkan cooperative matrices, with no floating GEMM fallback.

Like the TensorRT backend, this lowers only the canonical precision GEMM.
The original QDQ scales, clipping, ties-to-even rounding and operation count
are preserved. Quantization and output conversion execute on the GPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from onnx import numpy_helper

from .. import models
from ..stats import inference_samples, median, summarise
from .base import Backend, BackendError, CompiledModel

SHADERS = Path(__file__).resolve().parents[1] / "shaders"


def _spirv_int8_evidence(code):
    """Inspect the actual module passed to vkCreateShaderModule, not a flag.

    Opcodes follow Khronos SPIRV-Headers spirv.core.grammar.json. Every matrix
    multiply-add must take signed 8-bit A/B and signed 32-bit C/result matrices.
    """
    if len(code) % 4:
        raise BackendError("Invalid SPIR-V module length.")
    words = struct.unpack(f"<{len(code)//4}I", code)
    if len(words) < 5 or words[0] != 0x07230203:
        raise BackendError("Invalid SPIR-V header.")
    instructions = []
    index = 5
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 65535
        if not count or index + count > len(words):
            raise BackendError("Invalid SPIR-V instruction.")
        instructions.append((opcode, words[index+1:index+count]))
        index += count
    ints = {p[0]: (p[1], p[2]) for op, p in instructions if op == 21}  # OpTypeInt
    constants = {p[1]: p[2] for op, p in instructions if op == 43 and len(p) == 3}  # OpConstant
    matrices = {p[0]: (ints.get(p[1]), *(constants.get(i) for i in p[2:]))
                for op, p in instructions if op == 4456}  # OpTypeCooperativeMatrixKHR
    values = {p[1]: p[0] for op, p in instructions if len(p) >= 2 and p[0] in matrices}
    count = 0
    for op, p in instructions:
        if op != 4459:  # OpCooperativeMatrixMulAddKHR
            continue
        if (len(p) != 6 or p[5] != 15
                or matrices.get(values.get(p[2])) != ((8, 1), 3, 16, 16, 0)
                or matrices.get(values.get(p[3])) != ((8, 1), 3, 16, 16, 1)
                or matrices.get(values.get(p[4])) != ((32, 1), 3, 16, 16, 2)
                or matrices.get(p[0]) != ((32, 1), 3, 16, 16, 2)):
            raise BackendError("SPIR-V matrix operands/result do not prove signed INT8 / INT32 arithmetic.")
        count += 1
    if not count:
        raise BackendError("SPIR-V module contains no verified INT8 cooperative-matrix multiplication.")
    return {"opcode": "OpCooperativeMatrixMulAddKHR", "instructions": count,
            "operand_precision": "int8", "accumulator_precision": "int32"}


def _instance(v):
    return v.vkCreateInstance(v.VkInstanceCreateInfo(pApplicationInfo=v.VkApplicationInfo(
        pApplicationName="fairbench", apiVersion=v.VK_MAKE_VERSION(1, 3, 0))), None)


def _properties(v, instance, physical):
    query = v.vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")
    return [{name: int(getattr(p, name)) for name in (
        "MSize", "NSize", "KSize", "AType", "BType", "CType", "ResultType", "scope", "saturatingAccumulation")}
        for p in query(physical)]


def _int8_property(v, properties):
    return next((p for p in properties if (p["MSize"], p["NSize"], p["KSize"]) == (16, 16, 16)
                 and p["AType"] == p["BType"] == v.VK_COMPONENT_TYPE_SINT8_KHR
                 and p["CType"] == p["ResultType"] == v.VK_COMPONENT_TYPE_SINT32_KHR
                 and p["scope"] == v.VK_SCOPE_SUBGROUP_KHR and not p["saturatingAccumulation"]), None)


def probe_runtime() -> dict:
    if importlib.util.find_spec("vulkan") is None:
        return {}
    try:
        import vulkan as v
    except Exception as exc:
        return {"vulkan_error": f"{type(exc).__name__}: {exc}"}

    instance = None
    try:
        instance = _instance(v)
        devices = []
        for index, physical in enumerate(v.vkEnumeratePhysicalDevices(instance)):
            prop = v.vkGetPhysicalDeviceProperties(physical)
            extensions = {p.extensionName for p in v.vkEnumerateDeviceExtensionProperties(physical, None)}
            if (prop.vendorID != 0x1002 or prop.apiVersion < v.VK_MAKE_VERSION(1, 3, 0)
                    or "VK_KHR_cooperative_matrix" not in extensions):
                continue
            matrix = _int8_property(v, _properties(v, instance, physical))
            if matrix:
                devices.append({"device_id": index, "name": prop.deviceName,
                                "vendor_id": prop.vendorID, "pci_device_id": prop.deviceID,
                                "driver_version": prop.driverVersion, "int8_matrix": matrix})
        return {"vulkan_devices": devices, "vulkan_version": "1.3 cooperative matrices"}
    except Exception as exc:
        return {"vulkan_error": str(exc)}
    finally:
        if instance is not None:
            v.vkDestroyInstance(instance, None)


def _model_parameters(built, precision):
    if precision != "int8" or built.key != "precision_gemm_int8" or not built.gemm_shape:
        raise BackendError("Vulkan cooperative-matrix measurement supports precision_gemm_int8 only.")
    m, k, n = built.gemm_shape
    if any(d <= 0 or d > 4096 or d % 32 for d in (m, k, n)):
        raise BackendError("Vulkan INT8 GEMM requires dimensions divisible by 32 and at most 4096.")
    canonical = models.build_precision_gemm("int8", m if m == k == n else 0)
    if canonical.sha256 != built.sha256:
        raise BackendError("Vulkan INT8 lowering requires the unmodified canonical QDQ GEMM.")
    values = {t.name: numpy_helper.to_array(t) for t in built.model.graph.initializer}
    return (m, k, n), values


def _vector_grid(elements):
    groups = (elements + 255) // 256
    x = min(256, groups)
    return x, (groups + x - 1) // x


class _Execution:
    def __init__(self, built, target, values):
        import vulkan as v

        self.v = v
        self.resources = ExitStack()
        self.device = None
        self.shape = built.gemm_shape
        self.shader_hashes = {}
        try:
            instance = _instance(v)
            self.resources.callback(v.vkDestroyInstance, instance, None)
            physical = v.vkEnumeratePhysicalDevices(instance)[int(target.options["device_id"])]
            prop = v.vkGetPhysicalDeviceProperties(physical)
            if (prop.vendorID, prop.deviceID, prop.deviceName) != (
                    target.options["vendor_id"], target.options["pci_device_id"], target.detail):
                raise BackendError("Vulkan adapter changed; refresh device discovery.")
            self.matrix = _int8_property(v, _properties(v, instance, physical))
            self.device_info = {"name": prop.deviceName, "api_version": int(prop.apiVersion),
                                "driver_version": int(prop.driverVersion), "pci_device_id": int(prop.deviceID)}
            if not self.matrix:
                raise BackendError("Selected GPU does not expose signed INT8 / INT32 cooperative matrices.")
            qindex = next(i for i, p in enumerate(v.vkGetPhysicalDeviceQueueFamilyProperties(physical))
                          if p.queueFlags & v.VK_QUEUE_COMPUTE_BIT)
            features = v.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(cooperativeMatrix=True,
                pNext=v.VkPhysicalDevice8BitStorageFeatures(storageBuffer8BitAccess=True,
                pNext=v.VkPhysicalDeviceShaderFloat16Int8Features(shaderInt8=True,
                pNext=v.VkPhysicalDeviceSubgroupSizeControlFeatures(subgroupSizeControl=True, computeFullSubgroups=True,
                pNext=v.VkPhysicalDeviceVulkanMemoryModelFeatures(vulkanMemoryModel=True, vulkanMemoryModelDeviceScope=True)))))
            self.device = d = v.vkCreateDevice(physical, v.VkDeviceCreateInfo(pNext=features,
                queueCreateInfoCount=1, pQueueCreateInfos=[v.VkDeviceQueueCreateInfo(
                    queueFamilyIndex=qindex, queueCount=1, pQueuePriorities=[1.0])],
                enabledExtensionCount=1, ppEnabledExtensionNames=["VK_KHR_cooperative_matrix"]), None)
            self.resources.callback(v.vkDestroyDevice, d, None)
            self.queue = v.vkGetDeviceQueue(d, qindex, 0)
            self.memprops = v.vkGetPhysicalDeviceMemoryProperties(physical)
            m, k, n = self.shape
            self.buffers = [self._buffer(size, cached=i == 4)
                            for i, size in enumerate((4*m*k, k*n, m*k, 4*m*n, 4*m*n))]
            self.buffers[1][1][:] = values["weight"].tobytes()
            self.parameters = struct.pack("<IIIfff", m, k, n, float(values["input_scale"]),
                                          float(values["weight_scale"]), float(values["output_scale"]))
            bindings = [v.VkDescriptorSetLayoutBinding(binding=i, descriptorType=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=v.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(5)]
            layout = v.vkCreateDescriptorSetLayout(d, v.VkDescriptorSetLayoutCreateInfo(
                bindingCount=5, pBindings=bindings), None)
            self.resources.callback(v.vkDestroyDescriptorSetLayout, d, layout, None)
            pl = v.vkCreatePipelineLayout(d, v.VkPipelineLayoutCreateInfo(setLayoutCount=1, pSetLayouts=[layout],
                pushConstantRangeCount=1, pPushConstantRanges=[v.VkPushConstantRange(
                    stageFlags=v.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=len(self.parameters))]), None)
            self.resources.callback(v.vkDestroyPipelineLayout, d, pl, None)
            pipelines = []
            for name in ("int8_quantize", "int8_gemm", "int8_output"):
                code = (SHADERS / f"{name}.spv").read_bytes()
                if name == "int8_gemm":
                    self.spirv_evidence = _spirv_int8_evidence(code)
                self.shader_hashes[name] = hashlib.sha256(code).hexdigest()
                module = v.vkCreateShaderModule(d, v.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
                self.resources.callback(v.vkDestroyShaderModule, d, module, None)
                stage = v.VkPipelineShaderStageCreateInfo(stage=v.VK_SHADER_STAGE_COMPUTE_BIT,
                    module=module, pName="main", pNext=v.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo(
                        requiredSubgroupSize=32) if name == "int8_gemm" else None)
                pipeline = v.vkCreateComputePipelines(d, v.VK_NULL_HANDLE, 1,
                    [v.VkComputePipelineCreateInfo(stage=stage, layout=pl)], None)[0]
                pipelines.append(pipeline)
                self.resources.callback(v.vkDestroyPipeline, d, pipeline, None)
            pool = v.vkCreateDescriptorPool(d, v.VkDescriptorPoolCreateInfo(maxSets=1, poolSizeCount=1,
                pPoolSizes=[v.VkDescriptorPoolSize(type=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=5)]), None)
            self.resources.callback(v.vkDestroyDescriptorPool, d, pool, None)
            descriptors = v.vkAllocateDescriptorSets(d, v.VkDescriptorSetAllocateInfo(
                descriptorPool=pool, descriptorSetCount=1, pSetLayouts=[layout]))[0]
            v.vkUpdateDescriptorSets(d, 5, [v.VkWriteDescriptorSet(dstSet=descriptors, dstBinding=i,
                descriptorCount=1, descriptorType=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[v.VkDescriptorBufferInfo(buffer=b, offset=0, range=size)])
                for i, (b, data, size) in enumerate(self.buffers)], 0, None)
            cp = v.vkCreateCommandPool(d, v.VkCommandPoolCreateInfo(queueFamilyIndex=qindex), None)
            self.resources.callback(v.vkDestroyCommandPool, d, cp, None)
            self.command = cmd = v.vkAllocateCommandBuffers(d, v.VkCommandBufferAllocateInfo(commandPool=cp,
                level=v.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
            v.vkBeginCommandBuffer(cmd, v.VkCommandBufferBeginInfo())
            v.vkCmdBindDescriptorSets(cmd, v.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, [descriptors], 0, None)
            v.vkCmdPushConstants(cmd, pl, v.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(self.parameters),
                                 v.ffi.from_buffer(self.parameters))
            for pipeline, grid in zip(pipelines, (_vector_grid(m*k), (n//32, m//32), _vector_grid(m*n)), strict=True):
                v.vkCmdBindPipeline(cmd, v.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
                v.vkCmdDispatch(cmd, *grid, 1)
                v.vkCmdPipelineBarrier(cmd, v.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    v.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | v.VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                    [v.VkMemoryBarrier(srcAccessMask=v.VK_ACCESS_SHADER_WRITE_BIT,
                        dstAccessMask=v.VK_ACCESS_SHADER_READ_BIT | v.VK_ACCESS_HOST_READ_BIT)], 0, None, 0, None)
            v.vkEndCommandBuffer(cmd)
            self.fence = v.vkCreateFence(d, v.VkFenceCreateInfo(), None)
            self.resources.callback(v.vkDestroyFence, d, self.fence, None)
        except Exception:
            self.close()
            raise

    def _buffer(self, size, *, cached=False):
        v, d = self.v, self.device
        b = v.vkCreateBuffer(d, v.VkBufferCreateInfo(size=size, usage=v.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=v.VK_SHARING_MODE_EXCLUSIVE), None)
        try:
            req = v.vkGetBufferMemoryRequirements(d, b)
            candidates = [i for i in range(self.memprops.memoryTypeCount) if req.memoryTypeBits & (1 << i)
                          and self.memprops.memoryTypes[i].propertyFlags & 6 == 6]  # HOST_VISIBLE | HOST_COHERENT
            if not candidates:
                raise BackendError("GPU lacks coherent host-visible buffer memory.")
            # CPU reads of write-combined device memory can be hundreds of ms.
            # Put readback in HOST_CACHED memory; keep matrix operands device-local.
            preference = v.VK_MEMORY_PROPERTY_HOST_CACHED_BIT if cached else v.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            index = max(candidates, key=lambda i: bool(self.memprops.memoryTypes[i].propertyFlags & preference))
            memory = v.vkAllocateMemory(d, v.VkMemoryAllocateInfo(allocationSize=req.size, memoryTypeIndex=index), None)
        except Exception:
            v.vkDestroyBuffer(d, b, None)
            raise
        self.resources.callback(v.vkFreeMemory, d, memory, None)
        self.resources.callback(v.vkDestroyBuffer, d, b, None)
        v.vkBindBufferMemory(d, b, memory, 0)
        data = v.vkMapMemory(d, memory, 0, size, 0)
        self.resources.callback(v.vkUnmapMemory, d, memory)
        return b, data, size

    def upload(self, feed):
        value = feed.get("x")
        if set(feed) != {"x"} or value.dtype != np.float32 or value.shape != self.shape[:2] or not np.isfinite(value).all():
            raise BackendError("Vulkan input must be the finite FP32 canonical GEMM input.")
        np.copyto(np.frombuffer(self.buffers[0][1], dtype=np.float32).reshape(value.shape), value)

    def execute(self):
        v, d = self.v, self.device
        v.vkResetFences(d, 1, [self.fence])
        v.vkQueueSubmit(self.queue, 1, [v.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[self.command])], self.fence)
        v.vkWaitForFences(d, 1, [self.fence], True, 60_000_000_000)

    def download(self):
        return [np.frombuffer(self.buffers[4][1], dtype=np.float32).reshape(self.shape[0], self.shape[2]).copy()]

    def close(self):
        if not hasattr(self, "resources"):
            return
        if self.device is not None:
            try:
                self.v.vkDeviceWaitIdle(self.device)
            finally:
                self.device = None
                self.resources.close()
        else:
            self.resources.close()

    def __del__(self):
        self.close()


class VulkanBackend(Backend):
    id = "vulkan"
    name = "AMD Vulkan INT8"
    _IO_DEVICES = {"VulkanCooperativeMatrix": "vulkan"}

    @staticmethod
    def is_installed():
        return importlib.util.find_spec("vulkan") is not None

    @staticmethod
    def io_device(target):
        return "vulkan"

    def compile(self, built, target, cfg, precision):
        started = time.perf_counter()
        _, values = _model_parameters(built, precision)
        try:
            execution = _Execution(built, target, values)
        except Exception as exc:
            raise BackendError(f"Vulkan INT8 compilation failed; no fallback: {type(exc).__name__}: {exc}") from exc
        return CompiledModel(execution, (time.perf_counter()-started)*1000, precision, ["y"], {
            "execution_providers": [target.device], "requested_provider": target.device, "fallback_used": False,
            "cpu_fallback_disabled": True, "compute_precisions": ["int8"], "quantization": "static_qdq",
            "compute_precision_source": "Vulkan signed INT8 cooperative-matrix pipeline with INT32 accumulation",
            "shader_sha256": execution.shader_hashes, "cooperative_matrix": execution.matrix,
            "spirv_evidence": execution.spirv_evidence,
            "device": execution.device_info,
            "source_model_sha256": built.sha256, "buffer_memory": "host_visible_coherent_device_buffers",
            "gpu_precision": {"available": False, "arithmetic_precision": "int8", "accumulator_precision": "int32",
                "tensor_precisions": ["int8"], "evidence": "vulkan_cooperative_matrix_pipeline",
                "runtime_execution_verified": False,
                "note": "Signed INT8 matrix operands / INT32 accumulation declared in SPIR-V and supported by the selected GPU; no instruction disassembly."}})

    @staticmethod
    def _execute(compiled):
        compiled.handle.execute()
        compiled.meta["gpu_precision"].update(available=True, runtime_execution_verified=True)

    def run(self, compiled, feed):
        compiled.handle.upload(feed)
        self._execute(compiled)
        return compiled.handle.download()

    def prepare_resident(self, compiled, feed, target):
        compiled.handle.upload(feed)
        return lambda: self._execute(compiled), compiled.handle.download

    def diagnose(self, compiled, built, target, cfg, feed):
        phases = {"h2d_ms": [], "compute_ms": [], "d2h_ms": [], "e2e_ms": []}
        for _ in inference_samples(cfg, phases["compute_ms"]):
            started = time.perf_counter()
            compiled.handle.upload(feed)
            uploaded = time.perf_counter()
            self._execute(compiled)
            computed = time.perf_counter()
            compiled.handle.download()
            ended = time.perf_counter()
            for name, value in zip(phases, (uploaded-started, computed-uploaded, ended-computed, ended-started), strict=True):
                phases[name].append(value*1000)
        return {"available": True, "method": "synchronized_vulkan_fence_host_wall",
                "transfer_scope": "host_visible_coherent_device_buffers", "internal_transfers_in_compute": True,
                "samples": len(phases["compute_ms"]), "phase_samples_ms": phases,
                "compute_statistics": summarise(phases["compute_ms"]).to_dict(),
                **{name: median(values) for name, values in phases.items()},
                "note": "Inference includes GPU input quantization, INT8 GEMM, output requantization/dequantization, dispatch and fence synchronization. Host copies are separate."}
