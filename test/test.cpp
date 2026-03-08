//
// Created by muham on 21.02.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    auto train_df = ds::DataFrame::from_csv(R"(C:\software\Cpp\projects\CortexMind\test\archive\train2.csv)");
    train_df.info();

    auto [x_train, y_train] = train_df.split("y");

    float32 x_max = x_train.max();
    x_train *= 1.0f / x_max;

    net::Model model;
    model.add<nn::Dense>(3, 8);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(8, 1);
    model.compile<loss::MeanSquared, opt::StochasticGradient>(0.01f);
    model.callback<call::EarlyStopping>();
    model.summary();
    model.fit(x_train, y_train, max_epochs, -1);

    return cortex::exit;
}