#define INITGUID
#include <windows.h>
#include <dshow.h>
#include <ks.h>
#include <ksmedia.h>
#include <ksproxy.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "strmiids.lib")
#pragma comment(lib, "ksproxy.lib")

static const GUID EXTENDED_CAMERA_CONTROL =
    {0x1CB79112, 0xC0D2, 0x4213, {0x9C, 0xA6, 0xCD, 0x4F, 0xDB, 0x92, 0x79, 0x72}};

static void print_hr(const char* label, HRESULT hr) {
    std::cout << label << ": hr=0x"
              << std::hex << std::setw(8) << std::setfill('0') << hr
              << std::dec << std::setfill(' ') << "\n";
}

static std::wstring moniker_name(IMoniker* moniker) {
    IPropertyBag* bag = nullptr;
    VARIANT var;
    VariantInit(&var);

    if (FAILED(moniker->BindToStorage(nullptr, nullptr, IID_IPropertyBag, (void**)&bag))) {
        return L"(unknown)";
    }

    std::wstring name = L"(unknown)";
    if (SUCCEEDED(bag->Read(L"FriendlyName", &var, nullptr)) && var.vt == VT_BSTR) {
        name = var.bstrVal;
    }

    VariantClear(&var);
    bag->Release();
    return name;
}

static void print_bytes(const unsigned char* data, ULONG len) {
    std::cout << "    bytes=";
    for (ULONG i = 0; i < len && i < 160; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << (unsigned int)data[i] << " ";
    }
    std::cout << std::dec << std::setfill(' ') << "\n";
}

static void print_header(const KSCAMERA_EXTENDEDPROP_HEADER& header) {
    std::cout << "    Version=" << header.Version << "\n";
    std::cout << "    PinId=" << header.PinId << "\n";
    std::cout << "    Size=" << header.Size << "\n";
    std::cout << "    Result=0x" << std::hex << header.Result << std::dec << "\n";
    std::cout << "    Capability=0x" << std::hex << header.Capability << std::dec << "\n";
    std::cout << "    Flags=0x" << std::hex << header.Flags << std::dec << "\n";
}

static void decode_known_flags(ULONG id, const KSCAMERA_EXTENDEDPROP_HEADER& header) {
    if (id == 35) {
        std::cout << "    FaceAuth Capability:";
        if (header.Capability & 0x1) std::cout << " DISABLED";
        if (header.Capability & 0x2) std::cout << " ALTERNATIVE_FRAME_ILLUMINATION";
        if (header.Capability & 0x4) std::cout << " BACKGROUND_SUBTRACTION";
        std::cout << "\n";
        std::cout << "    FaceAuth Flags:";
        if (header.Flags & 0x1) std::cout << " DISABLED";
        if (header.Flags & 0x2) std::cout << " ALTERNATIVE_FRAME_ILLUMINATION";
        if (header.Flags & 0x4) std::cout << " BACKGROUND_SUBTRACTION";
        std::cout << "\n";
    } else if (id == 38) {
        std::cout << "    IRTorch Capability:";
        if (header.Capability & 0x1) std::cout << " OFF";
        if (header.Capability & 0x2) std::cout << " ALWAYS_ON";
        if (header.Capability & 0x4) std::cout << " ALTERNATING_FRAME_ILLUMINATION";
        std::cout << "\n";
        std::cout << "    IRTorch Flags:";
        if (header.Flags & 0x1) std::cout << " OFF";
        if (header.Flags & 0x2) std::cout << " ALWAYS_ON";
        if (header.Flags & 0x4) std::cout << " ALTERNATING_FRAME_ILLUMINATION";
        std::cout << "\n";
    }
}

static HRESULT ks_property(
    IKsControl* ks,
    ULONG id,
    ULONG flags,
    void* data,
    ULONG data_len,
    ULONG* returned
) {
    KSPROPERTY prop = {};
    prop.Set = EXTENDED_CAMERA_CONTROL;
    prop.Id = id;
    prop.Flags = flags;
    return ks->KsProperty(&prop, sizeof(prop), data, data_len, returned);
}

static void query_extended_property(IKsControl* ks, ULONG id, const char* name) {
    std::cout << "\n    property " << id << " " << name << "\n";

    ULONG returned = 0;
    BYTE basic[512] = {};
    HRESULT hr = ks_property(
        ks,
        id,
        KSPROPERTY_TYPE_BASICSUPPORT,
        basic,
        sizeof(basic),
        &returned
    );
    print_hr("      BASICSUPPORT", hr);
    if (SUCCEEDED(hr)) {
        std::cout << "      returned=" << returned << "\n";
        print_bytes(basic, returned);
    }

    KSCAMERA_EXTENDEDPROP_HEADER header = {};
    returned = 0;
    hr = ks_property(
        ks,
        id,
        KSPROPERTY_TYPE_GET,
        &header,
        sizeof(header),
        &returned
    );
    print_hr("      GET(header)", hr);
    if (SUCCEEDED(hr)) {
        std::cout << "      returned=" << returned << "\n";
        print_header(header);
        decode_known_flags(id, header);
    }

    std::vector<unsigned char> buf(1024);
    returned = 0;
    hr = ks_property(
        ks,
        id,
        KSPROPERTY_TYPE_GET,
        buf.data(),
        (ULONG)buf.size(),
        &returned
    );
    print_hr("      GET(1024)", hr);
    if (SUCCEEDED(hr)) {
        std::cout << "      returned=" << returned << "\n";
        print_bytes(buf.data(), returned);
    }
}

static void inspect_iks_control(IUnknown* object, const char* label) {
    IKsControl* ks = nullptr;
    HRESULT hr = object->QueryInterface(__uuidof(IKsControl), (void**)&ks);
    print_hr(label, hr);
    if (FAILED(hr)) {
        return;
    }

    query_extended_property(ks, 35, "FACEAUTH_MODE");
    query_extended_property(ks, 38, "IRTORCHMODE");

    ks->Release();
}

static void inspect_filter(IBaseFilter* filter) {
    inspect_iks_control(filter, "  filter QueryInterface(IKsControl)");

    IEnumPins* enum_pins = nullptr;
    HRESULT hr = filter->EnumPins(&enum_pins);
    if (FAILED(hr)) {
        return;
    }

    IPin* pin = nullptr;
    ULONG fetched = 0;
    int pin_index = 0;
    while (enum_pins->Next(1, &pin, &fetched) == S_OK) {
        PIN_INFO info = {};
        pin->QueryPinInfo(&info);
        std::wcout << L"\n  pin " << pin_index << L": "
                   << (info.achName[0] ? info.achName : L"(unnamed)") << L"\n";
        if (info.pFilter) {
            info.pFilter->Release();
        }

        inspect_iks_control(pin, "    pin QueryInterface(IKsControl)");

        pin->Release();
        pin_index++;
    }

    enum_pins->Release();
}

int wmain() {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        print_hr("CoInitializeEx", hr);
        return 1;
    }

    ICreateDevEnum* dev_enum = nullptr;
    hr = CoCreateInstance(
        CLSID_SystemDeviceEnum,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_ICreateDevEnum,
        (void**)&dev_enum
    );
    if (FAILED(hr)) {
        print_hr("CoCreateInstance(CLSID_SystemDeviceEnum)", hr);
        CoUninitialize();
        return 1;
    }

    IEnumMoniker* enum_moniker = nullptr;
    hr = dev_enum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enum_moniker, 0);
    if (hr != S_OK) {
        print_hr("CreateClassEnumerator(VideoInputDeviceCategory)", hr);
        dev_enum->Release();
        CoUninitialize();
        return 1;
    }

    IMoniker* moniker = nullptr;
    ULONG fetched = 0;
    int index = 0;
    while (enum_moniker->Next(1, &moniker, &fetched) == S_OK) {
        std::wstring name = moniker_name(moniker);
        std::wcout << L"\n[" << index << L"] " << name << L"\n";

        IBaseFilter* filter = nullptr;
        hr = moniker->BindToObject(nullptr, nullptr, IID_IBaseFilter, (void**)&filter);
        print_hr("  BindToObject(IBaseFilter)", hr);
        if (SUCCEEDED(hr)) {
            inspect_filter(filter);
            filter->Release();
        }

        moniker->Release();
        index++;
    }

    enum_moniker->Release();
    dev_enum->Release();
    CoUninitialize();
    return 0;
}
