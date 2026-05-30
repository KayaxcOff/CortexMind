//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    auto train_df = load(R"(..\test\archive\salary_elite.csv)");

    train_df.label_encode("branch");
    train_df.label_encode("company_type");
    train_df.label_encode("job_role");

    train_df["cgpa"].scale();
    train_df["coding_score"].scale();
    train_df["communication_score"].scale();
    train_df["aptitude_score"].scale();
    train_df["projects"].scale();
    train_df["resume_score"].scale();
    train_df["salary_lpa"].scale();

    train_df.drop("student_id");
    train_df.info();
    train_df.Set("salary_lpa");

    auto[x, y] = train_df.split();

    net::Model model;

    model.add<nn::Dense>(18, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(16, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 64);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(64, 64);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(64, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dropout>();
    model.add<nn::Dense>(16, 1);

    model.compile<loss::MeanSquared, opt::Adam>(0.0001f);
    model.summary();

    model.fit(x, y, 1000, 100);

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
<DataFrame: 9000 rows x 19 cols>
 - cgpa (float32)
 - branch (float32)
 - college_tier (float32)
 - python_skill (float32)
 - dsa_skill (float32)
 - ml_skill (float32)
 - web_dev_skill (float32)
 - coding_score (float32)
 - communication_score (float32)
 - aptitude_score (float32)
 - internships (float32)
 - projects (float32)
 - backlogs (float32)
 - resume_score (float32)
 - skill_score (float32)
 - placed (float32)
 - company_type (float32)
 - job_role (float32)
 - salary_lpa (float32)

==================================================
Model:
==================================================
Layer                         Mode
--------------------------------------------------
Dense(18, 16)                 Train
ReLU                          Train
Dense(16, 32)                 Train
ReLU                          Train
Dense(32, 64)                 Train
ReLU                          Train
Dense(64, 64)                 Train
ReLU                          Train
Dense(64, 32)                 Train
ReLU                          Train
Dense(32, 16)                 Train
ReLU                          Train
Dropout(0.100000)             Train
Dense(16, 1)                  Train
==================================================
Is compiled   : Yes
Loss Function : MSE
Optimizer     : Adam(0.000100)
Total Params  : 9745
==================================================
Epoch 0     | Loss: 0.187510%
Epoch 100   | Loss: 0.047994%
Epoch 200   | Loss: 0.023667%
Epoch 300   | Loss: 0.015905%
Epoch 400   | Loss: 0.012764%
Epoch 500   | Loss: 0.010260%
Epoch 600   | Loss: 0.009257%
Epoch 700   | Loss: 0.007889%
Epoch 800   | Loss: 0.007269%
Epoch 900   | Loss: 0.006760%

Process finished with exit code 0
*/