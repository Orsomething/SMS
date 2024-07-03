#pragma once  // 防止头文件重复包含

#include "CoreMinimal.h"  // 引入 Unreal Engine 的核心头文件
#include "Modules/ModuleManager.h"  // 引入模块管理器的头文件

class FSchedulingPluginModule : public IModuleInterface  // 定义插件模块类，实现 IModuleInterface 接口
{
public:
    virtual void StartupModule() override;  // 模块启动函数，需要在子类中实现
    virtual void ShutdownModule() override;  // 模块关闭函数，需要在子类中实现
};
