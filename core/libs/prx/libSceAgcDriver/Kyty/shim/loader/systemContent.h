#ifndef APS5_KYTY_SHIM_LOADER_SYSTEMCONTENT_H
#define APS5_KYTY_SHIM_LOADER_SYSTEMCONTENT_H

#include <string>

namespace Libs::Loader {

// TITLE_ID / CONTENT_ID of the running title (names the shader store); false when unknown.
bool SystemContentParamSfoGetString(const char* name, std::string* value);

} // namespace Libs::Loader

#endif
