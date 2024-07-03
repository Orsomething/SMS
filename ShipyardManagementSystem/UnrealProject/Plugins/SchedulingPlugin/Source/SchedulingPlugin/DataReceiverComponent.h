#pragma once  // 防止头文件重复包含

#include "CoreMinimal.h"  // 引入 Unreal Engine 的核心头文件
#include "Components/ActorComponent.h"  // 引入 ActorComponent 类的头文件
#include "DataReceiverComponent.generated.h"  // 生成的头文件，包含了类的元数据

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))  // 定义一个 Unreal 类
class SCHEDULINGPLUGIN_API UDataReceiverComponent : public UActorComponent  // 继承自 ActorComponent 类
{
    GENERATED_BODY()  // 自动生成类的声明和实现代码

public:
    UDataReceiverComponent();  // 构造函数声明
    virtual void BeginPlay() override;  // 重写 BeginPlay 函数

    UFUNCTION(BlueprintCallable, Category = "Data")  // 声明一个蓝图可调用的函数，属于 "Data" 类别
    void ReceiveData();  // 接收数据的函数声明

private:
    void ParsePlanningGoals(const FString& FilePath);  // 解析 Planning Goals 的函数声明
    void ParseSiteInfo(const FString& FilePath);  // 解析 Site Info 的函数声明
};
