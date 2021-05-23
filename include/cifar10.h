// Mimicing
// https://github.com/pytorch/pytorch/blob/abb215e22952ae44b764e501d3552bf219ceb95b/torch/csrc/api/include/torch/data/datasets/mnist.h
// https://pytorch.org/cppdocs/api/classtorch_1_1data_1_1datasets_1_1_m_n_i_s_t.html#class-mnist

#include <string>
#include <torch/torch.h>

class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10>
{
public:
    // The mode in which the dataset is loaded.
    enum class Mode
    {
        kTrain,
        kTest
    };

    explicit CIFAR10(const std::string& root, Mode mode = Mode::kTrain);

    // https://pytorch.org/cppdocs/api/structtorch_1_1data_1_1_example.html#structtorch_1_1data_1_1_example
    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

    bool is_train() const noexcept;

    // Returns all images stacked into a single tensor.
    const torch::Tensor& images() const;

    const torch::Tensor& targets() const;

private:
    // Returns all targets stacked into a single tensor.
    torch::Tensor images_, targets_;
};
