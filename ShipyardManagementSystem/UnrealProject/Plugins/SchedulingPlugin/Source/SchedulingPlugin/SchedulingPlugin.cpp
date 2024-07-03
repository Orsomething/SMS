#include "SchedulingPlugin.h"  // 引入插件的头文件
#include "Modules/ModuleManager.h"  // 引入模块管理器的头文件
#include "DataReceiverComponent.h"  // 引入数据接收组件的头文件
#include "DataSenderComponent.h"  // 引入数据发送组件的头文件

IMPLEMENT_MODULE(FSchedulingPluginModule, SchedulingPlugin)  // 实现插件模块

void FSchedulingPluginModule::StartupModule() {
    // 在模块启动时的初始化逻辑
    // 可以在这里进行插件的初始化操作
}

void FSchedulingPluginModule::ShutdownModule() {
    // 在模块关闭时的清理逻辑
    // 可以在这里进行插件的清理操作
}
