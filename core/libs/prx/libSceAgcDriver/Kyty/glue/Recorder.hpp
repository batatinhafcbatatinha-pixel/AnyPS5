#ifndef APS5_KYTY_GLUE_RECORDER_HPP
#define APS5_KYTY_GLUE_RECORDER_HPP

namespace AgcDriver::VramBackend::Recorder {

// After the device dispatcher is loaded: GPU-thread command buffers are recorded on their own
// thread (APS5_KYTY_RECORDER=0: off). See Recorder.cpp.
void Install();

} // namespace AgcDriver::VramBackend::Recorder

#endif
