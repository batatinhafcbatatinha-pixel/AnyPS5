#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTPLANBUILDER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTPLANBUILDER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler::Detail {

class PlanBuilder {
public:
    // imagesOnly (APS5_GPU_SRT=3): plan only the reads image and sampler descriptors depend on;
    // every other SRT read stays a run-time BDA read.
    explicit PlanBuilder(IrProgram& program, bool imagesOnly = false) : _program(program), _imagesOnly(imagesOnly) {}

    void Run();

private:
    struct Patch {
        IrValue* inst = nullptr;
        std::uint32_t slot = 0;
        bool keep = false;
    };

    void Collect(IrValue* raw, std::uint32_t usePc);
    void Record(IrValue* inst);
    void PatchReads();

    IrProgram& _program;
    bool _imagesOnly = false;
    std::vector<IrValue*> _visiting;
    std::unordered_map<IrValue*, std::size_t> _visitingIndex;
    std::unordered_set<IrValue*> _visited;
    std::vector<Patch> _patches;
};

}

#endif
