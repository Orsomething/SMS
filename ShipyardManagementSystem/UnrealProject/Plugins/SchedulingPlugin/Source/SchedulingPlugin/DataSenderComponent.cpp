#include "DataSenderComponent.h"  // 引入 DataSenderComponent 类的头文件
#include "Misc/FileHelper.h"  // 引入文件操作相关的头文件
#include "Misc/Paths.h"  // 引入路径操作相关的头文件

UDataSenderComponent::UDataSenderComponent() {
    PrimaryComponentTick.bCanEverTick = false;  // 设置组件不需要每帧 Tick
}

void UDataSenderComponent::BeginPlay() {
    Super::BeginPlay();  // 调用父类的 BeginPlay 函数
    SendScheduleResult();  // 在开始播放时调用发送调度结果函数
}

void UDataSenderComponent::SendScheduleResult() {
    // 构建 schedule_result.json 文件的路径
    FString ScheduleResultPath = FPaths::ProjectDir() / TEXT("data/output/schedule_result.json");
    FString ResultData = TEXT("{ \"plan\": \"Optimal plan from Scheduler\" }");  // 设置调度结果数据

    // 写入调度结果数据到文件
    WriteScheduleResult(ScheduleResultPath, ResultData);
}

void UDataSenderComponent::WriteScheduleResult(const FString& FilePath, const FString& ResultData) {
    if (FFileHelper::SaveStringToFile(ResultData, *FilePath)) {  // 将数据保存到文件
        UE_LOG(LogTemp, Warning, TEXT("Saved Schedule Result: %s"), *ResultData);  // 输出保存成功的日志
    }
}
