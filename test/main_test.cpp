//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {

    auto train_df = load(R"(..\test\archive\Teen_Mental_Health_Dataset.csv)");
    train_df.Set("depression_label");

    train_df.label_encode("gender");
    train_df.label_encode("platform_usage");
    train_df.label_encode("social_interaction_level");

    train_df["age"].scale();
    train_df["daily_social_media_hours"].scale();
    train_df["sleep_hours"].scale();
    train_df["screen_time_before_sleep"].scale();
    train_df["academic_performance"].scale();
    train_df["physical_activity"].scale();
    train_df["stress_level"].scale();
    train_df["anxiety_level"].scale();
    train_df["addiction_level"].scale();

    auto[x, y] = train_df.split();

    net::Model model;

    model.add<nn::Dense>(12, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 1);
    model.add<nn::Sigmoid>();

    model.compile<loss::BinaryCrossEntropy, opt::Adam, metric::Accuracy>();
    model.fit(x, y, 1000, 100);

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch     0 | Loss: 0.684001% | Metric: 0.529167
Epoch   100 | Loss: 0.121951% | Metric: 0.974167
Epoch   200 | Loss: 0.098952% | Metric: 0.974167
Epoch   300 | Loss: 0.073791% | Metric: 0.974167
Epoch   400 | Loss: 0.052781% | Metric: 0.974167
Epoch   500 | Loss: 0.036998% | Metric: 0.977500
Epoch   600 | Loss: 0.026367% | Metric: 0.990833
Epoch   700 | Loss: 0.018714% | Metric: 0.995000
Epoch   800 | Loss: 0.013183% | Metric: 0.995833
Epoch   900 | Loss: 0.009646% | Metric: 0.998333

Process finished with exit code 0
*/