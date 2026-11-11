#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SHADERINFOCOLLECTOR_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SHADERINFOCOLLECTOR_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include <cstdint>
#include <map>
#include <string>

namespace ShaderRecompiler {

class ShaderInfoCollector {
public:
    void Collect(IrProgram& program) const;
    void Collect(IrProgram& program, const ShaderStageInputInfo& inputInfo) const;
};

[[nodiscard]] bool ProveGridPointwise(const IrProgram& program);
[[nodiscard]] std::map<std::string, std::uint32_t> GridPointwiseFailures(const IrProgram& program);

}

#endif
