#include <memory>
#include <torch/torch.h>

struct BasicBlock : torch::nn::Module
{
    BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
               torch::nn::Sequential downsample = torch::nn::Sequential(),
               int64_t groups = 1, int64_t base_width = 64,
               int64_t dilation = 1);

    static const int64_t m_expansion = 1;

    torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr}, m_bn2{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::Sequential m_downsample = torch::nn::Sequential();

    int64_t m_stride;

    torch::Tensor forward(torch::Tensor x);
};

struct Bottleneck : torch::nn::Module
{
    Bottleneck(int64_t inplanes, int64_t planes, int64_t stride = 1,
               torch::nn::Sequential downsample = torch::nn::Sequential(),
               int64_t groups = 1, int64_t base_width = 64,
               int64_t dilation = 1);

    static const int64_t m_expansion = 4;

    torch::nn::Conv2d m_conv1{nullptr}, m_conv2{nullptr}, m_conv3{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr}, m_bn2{nullptr}, m_bn3{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::Sequential m_downsample = torch::nn::Sequential();

    int64_t m_stride;

    torch::Tensor forward(torch::Tensor x);
};

template <typename Block>
struct ResNet : torch::nn::Module
{
    ResNet(const std::vector<int64_t> layers, int64_t num_classes = 1000,
           bool zero_init_residual = false, int64_t groups = 1,
           int64_t width_per_group = 64,
           std::vector<int64_t> replace_stride_with_dilation = {});

    int64_t m_inplanes = 64;
    int64_t m_dilation = 1;
    int64_t m_groups = 1;
    int64_t m_base_width = 64;

    torch::nn::Conv2d m_conv1{nullptr};
    torch::nn::BatchNorm2d m_bn1{nullptr};
    torch::nn::ReLU m_relu{nullptr};
    torch::nn::MaxPool2d m_maxpool{nullptr};
    torch::nn::Sequential m_layer1{nullptr}, m_layer2{nullptr},
        m_layer3{nullptr}, m_layer4{nullptr};
    torch::nn::AdaptiveAvgPool2d m_avgpool{nullptr};
    torch::nn::Linear m_fc{nullptr};

    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks,
                                      int64_t stride = 1, bool dilate = false);

    torch::Tensor _forward_impl(torch::Tensor x);

    torch::Tensor forward(torch::Tensor x);
};

template <class Block>
std::shared_ptr<ResNet<Block>>
_resnet(const std::vector<int64_t>& layers, int64_t num_classes = 1000,
        bool zero_init_residual = false, int64_t groups = 1,
        int64_t width_per_group = 64,
        const std::vector<int64_t>& replace_stride_with_dilation = {});

std::shared_ptr<ResNet<BasicBlock>>
resnet18(int64_t num_classes = 1000, bool zero_init_residual = false,
         int64_t groups = 1, int64_t width_per_group = 64,
         std::vector<int64_t> replace_stride_with_dilation = {});

std::shared_ptr<ResNet<BasicBlock>>
resnet34(int64_t num_classes = 1000, bool zero_init_residual = false,
         int64_t groups = 1, int64_t width_per_group = 64,
         std::vector<int64_t> replace_stride_with_dilation = {});

std::shared_ptr<ResNet<Bottleneck>>
resnet50(int64_t num_classes = 1000, bool zero_init_residual = false,
         int64_t groups = 1, int64_t width_per_group = 64,
         std::vector<int64_t> replace_stride_with_dilation = {});

std::shared_ptr<ResNet<Bottleneck>>
resnet101(int64_t num_classes = 1000, bool zero_init_residual = false,
          int64_t groups = 1, int64_t width_per_group = 64,
          std::vector<int64_t> replace_stride_with_dilation = {});

std::shared_ptr<ResNet<Bottleneck>>
resnet152(int64_t num_classes = 1000, bool zero_init_residual = false,
          int64_t groups = 1, int64_t width_per_group = 64,
          std::vector<int64_t> replace_stride_with_dilation = {});

std::shared_ptr<ResNet<Bottleneck>>
resnext50_32x4d(int64_t num_classes = 1000, bool zero_init_residual = false,
                int64_t groups = 1, int64_t width_per_group = 64,
                std::vector<int64_t> replace_stride_with_dilation = {});

std::shared_ptr<ResNet<Bottleneck>>
resnext101_32x8d(int64_t num_classes = 1000, bool zero_init_residual = false,
                 int64_t groups = 1, int64_t width_per_group = 64,
                 std::vector<int64_t> replace_stride_with_dilation = {});

std::shared_ptr<ResNet<Bottleneck>>
wide_resnet50_2(int64_t num_classes = 1000, bool zero_init_residual = false,
                int64_t groups = 1, int64_t width_per_group = 64,
                std::vector<int64_t> replace_stride_with_dilation = {});

std::shared_ptr<ResNet<Bottleneck>>
wide_resnet101_2(int64_t num_classes = 1000, bool zero_init_residual = false,
                 int64_t groups = 1, int64_t width_per_group = 64,
                 std::vector<int64_t> replace_stride_with_dilation = {});