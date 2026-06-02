//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    auto train_df = load(R"(..\test\archive\winequality-white new.csv)");
    train_df.Set("pH");
    train_df.drop("alcohol");

    train_df["fixed acidity"].scale();
    train_df["volatile acidity"].scale();
    train_df["citric acid"].scale();
    train_df["residual sugar"].scale();
    train_df["chlorides"].scale();
    train_df["free sulfur dioxide"].scale();
    train_df["total sulfur dioxide"].scale();
    train_df["density"].scale();
    train_df["pH"].scale();
    train_df["sulphates"].scale();
    train_df["quality"].scale();

    auto[x, y] = train_df.split();

    net::Model model;

    model.add<nn::Dense>(10, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(32, 32);
    model.add<nn::ReLU>();
    model.add<nn::Dropout>();
    model.add<nn::Dense>(32, 16);
    model.add<nn::ReLU>();
    model.add<nn::Dense>(16, 1);
    model.compile<loss::MeanSquared, opt::Adam, metric::RootMeanSquared>();

    model.fit(x, y, 100, 10);

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
Epoch     0 | Loss: 0.221528 | RMSE: 0.470667
Epoch    10 | Loss: 0.122767 | RMSE: 0.350382
Epoch    20 | Loss: 0.056801 | RMSE: 0.238331
Epoch    30 | Loss: 0.026964 | RMSE: 0.164208
Epoch    40 | Loss: 0.027557 | RMSE: 0.166004
Epoch    50 | Loss: 0.023533 | RMSE: 0.153403
Epoch    60 | Loss: 0.021683 | RMSE: 0.147253
Epoch    70 | Loss: 0.020617 | RMSE: 0.143587
Epoch    80 | Loss: 0.019715 | RMSE: 0.140410
Epoch    90 | Loss: 0.018866 | RMSE: 0.137354

Process finished with exit code 0
*/