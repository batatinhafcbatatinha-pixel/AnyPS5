#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <cstdio>

namespace AgcDriver::Graphics {

VkDeviceAddress Buffer::DeviceAddress() const {
    if (deviceAddress == 0) {
        char text[160];
        std::snprintf(text, sizeof(text), "buffer has no device address (usage 0x%x, size %zu, caller %p)", static_cast<unsigned>(usage), size, __builtin_return_address(0));
        Require(false, text);
    }
    return deviceAddress;
}

void Buffer::initializeAddress(VkBufferUsageFlags usage) {
    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) == 0) return;
    const VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, buffer};
    deviceAddress = context.Function<PFN_vkGetBufferDeviceAddressKHR>("vkGetBufferDeviceAddressKHR")(context.device, &info);
    Require(deviceAddress != 0, "vkGetBufferDeviceAddressKHR returned a null address");
}

}
