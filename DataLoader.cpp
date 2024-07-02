#include "DataLoader.h"
#include <fstream>
#include <nlohmann/json.hpp> // 使用JSON for Modern C++库进行JSON解析

using json = nlohmann::json;

// 从文件加载规划目标
std::vector<PlanningGoal> DataLoader::loadPlanningGoals(const std::string& filepath) {
    std::ifstream file(filepath);
    json j;
    file >> j;

    std::vector<PlanningGoal> goals;
    for (const auto& item : j) {
        PlanningGoal goal;
        goal.goal = item["goal"];
        // 解析其他字段
        goals.push_back(goal);
    }

    return goals;
}

// 从文件加载场地信息
SiteInfo DataLoader::loadSiteInfo(const std::string& filepath) {
    std::ifstream file(filepath);
    json j;
    file >> j;

    SiteInfo siteInfo;
    siteInfo.info = j["info"];
    // 解析其他字段

    return siteInfo;
}

// 保存调度结果到文件
void DataLoader::saveSchedule(const Schedule& schedule, const std::string& filepath) {
    json j;
    j["plan"] = schedule.plan;
    // 添加其他字段

    std::ofstream file(filepath);
    file << j.dump(4); // 将JSON对象序列化并保存到文件，缩进为4个空格
}
