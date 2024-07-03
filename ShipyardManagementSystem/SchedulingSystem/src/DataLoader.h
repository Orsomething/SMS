#pragma once
#include <vector>
#include <string>
#include "Scheduler.h"

class DataLoader {
public:
    std::vector<PlanningGoal> loadPlanningGoals(const std::string& filepath);
    SiteInfo loadSiteInfo(const std::string& filepath);
    void saveSchedule(const Schedule& schedule, const std::string& filepath);
};
