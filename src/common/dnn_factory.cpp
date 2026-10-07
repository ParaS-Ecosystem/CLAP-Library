#include "clap/dnn_factory.hpp"
#include "clap/cudnn_backend.hpp"
#include "clap/dnn_dyn_backends.hpp"
#include "clap/miopen_backend.hpp"
#include "clap/onednn_backend.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <fstream>

namespace clap {

std::unique_ptr<IDnnBackend> DnnFactory::create(DnnBackendType requested)
{
    if (const char* env = std::getenv("CLAP_BACKEND")) {
        const std::string s(env);
        if (s == "CPU") requested = DnnBackendType::CPU;
        else if (s == "CUDA") requested = DnnBackendType::CUDA;
        else if (s == "ROCM" || s == "AMD") requested = DnnBackendType::ROCM;
        else if (s == "GPU") requested = DnnBackendType::GPU;
    }
     
    else {
	// 2. Check command-line arguments natively by reading process memory (Linux specific)
        std::ifstream cmdline("/proc/self/cmdline");
        if (cmdline.is_open()) {
            std::string arg;
            // Arguments in cmdline are separated by null bytes ('\0')
            while (std::getline(cmdline, arg, '\0')) {
                if (arg == "-cuda") { requested = DnnBackendType::CUDA; break; }
                if (arg == "-rocm") { requested = DnnBackendType::ROCM; break; }
                if (arg == "-cpu")  { requested = DnnBackendType::CPU;  break; }
                if (arg == "-gpu")  { requested = DnnBackendType::GPU;  break; }
          }
	 }
	}

    if (requested == DnnBackendType::CPU) {
        if (!dnn_dyn::loadOneDnn())
            throw std::runtime_error(std::string("CLAP_DNN oneDNN runtime load failed: ") + dnn_dyn::lastError());
        return std::make_unique<OneDnnBackend>();
    }

    if (requested == DnnBackendType::CUDA) {
        if (!dnn_dyn::loadCudaAndCudnn())
            throw std::runtime_error(std::string("CLAP_DNN CUDA/cuDNN runtime load failed: ") + dnn_dyn::lastError());
        return std::make_unique<CuDnnBackend>();
    }

    if (requested == DnnBackendType::ROCM) {
        if (!dnn_dyn::loadHipAndMiopen())
            throw std::runtime_error(std::string("CLAP_DNN HIP/MIOpen runtime load failed: ") + dnn_dyn::lastError());
        return std::make_unique<MiOpenBackend>();
    }

    // -gpu => automatic runtime vendor selection.
    if (dnn_dyn::loadCudaAndCudnn())
        return std::make_unique<CuDnnBackend>();

    if (dnn_dyn::loadHipAndMiopen())
        return std::make_unique<MiOpenBackend>();

    throw std::runtime_error("CLAP_DNN -gpu selected, but neither NVIDIA+cuDNN nor AMD+MIOpen could be loaded at runtime");
}

} // namespace clap
