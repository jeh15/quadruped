#include <filesystem>
#include <iostream>
#include <numeric>

#include "rules_cc/cc/runfiles/runfiles.h"

#include <onnxruntime_cxx_api.h>


using rules_cc::cc::runfiles::Runfiles;


template <typename T>
T vector_product(const std::vector<T>& v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

int main(int argc, char** argv) {
    // Use Runfiles to get the path to the model:
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path model_path = 
        runfiles->Rlocation("unitree-interface/onnx_models/breezy-dawn-149.onnx");

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeTest");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, model_path.c_str(), session_options);
    
    // Initialize Inputs and Outputs:
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> input_nodes;
    std::vector<Ort::AllocatedStringPtr> output_nodes;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<ONNXTensorElementDataType> output_types;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;

    // Get Input Names and Shapes
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        // Get Input Names:
        input_nodes.push_back(session.GetInputNameAllocated(i, allocator));
        input_names.push_back(input_nodes.back().get());
        // Get Input Types and Shapes:
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_types.push_back(tensor_info.GetElementType());
        input_shapes.push_back(tensor_info.GetShape());
    }

    // Get Output Names and Shapes
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        // Get Output Names:
        output_nodes.push_back(session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_nodes.back().get());
        // Get Output Types and Shapes:
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_types.push_back(tensor_info.GetElementType());
        output_shapes.push_back(tensor_info.GetShape());
    }
    
    // Print Input and Output Names and Shapes
    std::cout << "Input Names: " << std::endl;
    for (const auto& name : input_names) {
        std::cout << name << std::endl;
    }

    std::cout << "Input Types: " << std::endl;
    for (const auto& type : input_types) {
        std::cout << type << std::endl;
    }

    std::cout << "Input Shapes: " << std::endl;
    for (const auto& shape : input_shapes) {
        std::cout << "[";
        for (const auto& dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "Output Names: " << std::endl;
    for (const auto& name : output_names) {
        std::cout << name << std::endl;
    }

    std::cout << "Output Types: " << std::endl;
    for (const auto& type : output_types) {
        std::cout << type << std::endl;
    }
    
    std::cout << "Output Shapes: " << std::endl;
    for (const auto& shape : output_shapes) {
        std::cout << "[";
        for (const auto& dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << "]" << std::endl;
    }

    // Initialize Input and Output Vectors:
    size_t input_tensor_size = vector_product(input_shapes[0]);
    size_t output_tensor_size = vector_product(output_shapes[0]);
    std::vector<float> policy_input;
    std::vector<float> policy_output;
    policy_input.resize(input_tensor_size);
    policy_output.resize(output_tensor_size);

    // Initialize Input and Output Tensors:
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        policy_input.data(),
        policy_input.size(),
        input_shapes[0].data(),
        input_shapes[0].size()
    );

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        policy_output.data(),
        policy_output.size(),
        output_shapes[0].data(),
        output_shapes[0].size()
    );

    // Inference:
    Ort::RunOptions run_options;
    session.Run(
        run_options,
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        &output_tensor,
        1
    );


    // Check Input Tensor:
    if (!input_tensor.HasValue()) {
        std::cerr << "Failed to get input tensor" << std::endl;
        return -1;
    }
    std::cout << "Input Tensor: " << std::endl;
    for (size_t i = 0; i < input_tensor_size; ++i) {
        std::cout << policy_input[i] << " ";
    }
    std::cout << std::endl;

    // Check Output Tensor:
    if (!output_tensor.HasValue()) {
        std::cerr << "Failed to get output tensor" << std::endl;
        return -1;
    }
    std::cout << "Output Tensor: " << std::endl;
    for (size_t i = 0; i < output_tensor_size; ++i) {
        std::cout << policy_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}