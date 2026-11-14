#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTPLANBUILDER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTPLANBUILDER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler::Detail {

// Whether a program stores to or atomically updates any image (APS5_GPU_SRT=4 keeps its images bound).
bool ProgramWritesImages(const IrProgram& program);

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

    bool DependsOnShaderBase(const IrValue* value) const;
    void FindShaderBased();
    std::unordered_set<const IrValue*> _shaderBased;
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
