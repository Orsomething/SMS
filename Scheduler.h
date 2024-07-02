#pragma once

#include <vector>
#include <string>

// 规划目标结构体，包含目标描述和可能的其他字段
struct PlanningGoal {
    std::string goal;
    // 可以添加其他字段...
};

// 场地信息结构体，包含信息描述和可能的其他字段
struct SiteInfo {
    std::string info;
    // 可以添加其他字段...
};

// 调度结果结构体，包含计划描述和可能的其他字段
struct Schedule {
    std::string plan;
    // 可以添加其他字段...
};

// 调度器类，用于生成调度结果
class Scheduler {
public:
    // 创建调度结果的方法，接收规划目标和场地信息作为输入
    Schedule createSchedule(const std::vector<PlanningGoal>& goals, const SiteInfo& siteInfo);
};
