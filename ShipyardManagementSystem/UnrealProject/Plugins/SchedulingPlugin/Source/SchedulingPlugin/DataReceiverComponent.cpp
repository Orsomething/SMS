#include "DataReceiverComponent.h"  // 引入 DataReceiverComponent 类的头文件
#include "Misc/FileHelper.h"  // 引入文件操作相关的头文件
#include "Misc/Paths.h"  // 引入路径操作相关的头文件

UDataReceiverComponent::UDataReceiverComponent() {
    PrimaryComponentTick.bCanEverTick = false;  // 设置组件不需要每帧 Tick
}

void UDataReceiverComponent::BeginPlay() {
    Super::BeginPlay();  // 调用父类的 BeginPlay 函数
    ReceiveData();  // 在开始播放时调用接收数据函数
}

void UDataReceiverComponent::ReceiveData() {
    // 构建 planning_goals.json 和 site_info.json 文件的路径
    FString PlanningGoalsPath = FPaths::ProjectDir() / TEXT("data/input/planning_goals.json");
    FString SiteInfoPath = FPaths::ProjectDir() / TEXT("data/input/site_info.json");

    // 解析 planning_goals.json 和 site_info.json 文件
    ParsePlanningGoals(PlanningGoalsPath);
    ParseSiteInfo(SiteInfoPath);
}

void UDataReceiverComponent::ParsePlanningGoals(const FString& FilePath) {
    FString FileContent;
    if (FFileHelper::LoadFileToString(FileContent, *FilePath)) {
        // 如果成功加载文件内容，则输出日志
        UE_LOG(LogTemp, Warning, TEXT("Loaded Planning Goals: %s"), *FileContent);
    }
}

void UDataReceiverComponent::ParseSiteInfo(const FString& FilePath) {
    FString FileContent;
    if (FFileHelper::LoadFileToString(FileContent, *FilePath)) {
        // 如果成功加载文件内容，则输出日志
        UE_LOG(LogTemp, Warning, TEXT("Loaded Site Info: %s"), *FileContent);
    }
}
