#pragma once

#include <vector>
#include <string>
#include "Scheduler.h"

// 数据加载器类，用于加载规划目标、场地信息和保存调度结果
class DataLoader {
public:
    // 加载规划目标的方法，从指定文件路径加载数据并返回一个规划目标的向量
    std::vector<PlanningGoal> loadPlanningGoals(const std::string& filepath);

    // 加载场地信息的方法，从指定文件路径加载数据并返回场地信息
    SiteInfo loadSiteInfo(const std::string& filepath);

    // 保存调度结果的方法，将调度结果保存到指定文件路径
    void saveSchedule(const Schedule& schedule, const std::string& filepath);
};
