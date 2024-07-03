#include "Scheduler.h" // 包含调度器类的头文件
#include "DataLoader.h" // 包含数据加载器类的头文件

int main() {
    DataLoader dataLoader; // 创建数据加载器对象
    auto planningGoals = dataLoader.loadPlanningGoals("data/input/planning_goals.json"); // 加载规划目标数据
    auto siteInfo = dataLoader.loadSiteInfo("data/input/site_info.json"); // 加载场地信息数据

    Scheduler scheduler; // 创建调度器对象
    auto schedule = scheduler.createSchedule(planningGoals, siteInfo); // 生成调度结果

    dataLoader.saveSchedule(schedule, "data/output/schedule_result.json"); // 保存调度结果到文件

    return 0;
}
