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
Epoch     0 | Loss: 0.164955 | RMSE: 0.406147
Epoch    10 | Loss: 0.069818 | RMSE: 0.264232
Epoch    20 | Loss: 0.029073 | RMSE: 0.170508
Epoch    30 | Loss: 0.029341 | RMSE: 0.171293
Epoch    40 | Loss: 0.024132 | RMSE: 0.155344
Epoch    50 | Loss: 0.023103 | RMSE: 0.151997
Epoch    60 | Loss: 0.021696 | RMSE: 0.147297
Epoch    70 | Loss: 0.020721 | RMSE: 0.143949
Epoch    80 | Loss: 0.020681 | RMSE: 0.143808
Epoch    90 | Loss: 0.020203 | RMSE: 0.142136

Process finished with exit code 0
*/