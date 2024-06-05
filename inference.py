# import torch
# import torch.nn as nn
# import numpy as np
# import tensorflow as tf

# # Define the ReskinModel class for PyTorch
# class ReskinModel(nn.Module):
#     def __init__(self):
#         super(ReskinModel, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(15, 200),
#             nn.ReLU(),
#             nn.Dropout(p=0.15),
#             nn.Linear(200, 200),
#             nn.Linear(200, 40),
#             nn.Linear(40, 200),
#             nn.ReLU(),
#             nn.Dropout(p=0.15),
#             nn.Linear(200, 200),
#             nn.ReLU(),
#             nn.Linear(200, 3)
#         )

#     def forward(self, x):
#         return self.model(x)

# # Function to perform inference with PyTorch model
# def infer_with_pytorch_model(model_path, input_data):
#     model = ReskinModel()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # Set the model to evaluation mode

#     input_tensor = torch.tensor(input_data)
#     with torch.no_grad():
#         output = model(input_tensor)
#     return output.numpy()

# # Function to perform inference with TFLite model
# def infer_with_tflite_model(model_path, input_data):
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()

#     output = interpreter.get_tensor(output_details[0]['index'])
#     return output

# denorm_vecB = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 
#                             0.1, 0.1, 0.1, 0.1, 0.1, 
#                             0.1, 0.1, 0.1, 0.1, 0.1])
# denorm_vecF = np.array([10, 10, -4])


# # Generate random input data
# # random_input = np.random.rand(1, 15).astype(np.float32)
# test_input = np.array([78.3,-65.7,102.85,-54.45,-8.85,95.83,77.4,46.65,118.34,-119.25,-77.4,88.81,76.95,-129.45,85.67],dtype=np.float32)
# test_input_norm = test_input*denorm_vecB

# # Perform inference with both models
# pytorch_output = infer_with_pytorch_model("reskin_model.pt", test_input_norm)
# tflite_output = infer_with_tflite_model("reskin_model.tflite", test_input_norm)

# # Print the results
# print("PyTorch model prediction:", pytorch_output*denorm_vecF)
# print("TFLite model prediction:", tflite_output*denorm_vecF)


import torch
import torch.nn as nn
import numpy as np
# import tensorflow as tf
from train_lightning import IndentDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset
from matplotlib import pyplot as plt

# Define the ReskinModel class for PyTorch
class ReskinModel(nn.Module):
    def __init__(self):
        super(ReskinModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(15, 200),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(200, 200),
            nn.Linear(200, 40),
            nn.Linear(40, 200),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 3)
        )

    def forward(self, x):
        return self.model(x)

# Function to perform inference with PyTorch model
def infer_with_pytorch_model(model_path, input_data):
    model = ReskinModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    return output.numpy()

# Function to perform inference with TFLite model
# def infer_with_tflite_model(model_path, input_data):
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Ensure input data is float32 and reshape to 2D
#     input_data = input_data.astype(np.float32).reshape(1, -1)

#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()

#     output = interpreter.get_tensor(output_details[0]['index'])
#     return output

fpath = ['./combined_data_20240603_171745.csv', './combined_data_small.csv']
datasets_list = []
for raw_data_path in fpath:
    datasets_list.append(IndentDataset(raw_data_path, skip=0))
    print("Loaded dataset length: ",len(datasets_list[-1]))

batch_size = 1
train_dataset = ConcatDataset(datasets_list[:-1])
# train_dataset, test_dataset = random_split(full_dataset, [int(len(full_dataset)*0.9), len(full_dataset)-int(len(full_dataset)*0.9)])
test_dataset = datasets_list[-1]
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_data_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

denorm_vecB = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 
                        0.1, 0.1, 0.1, 0.1, 0.1, 
                        0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
denorm_vecF = np.array([10, 10, -4])

# Generate random input data
# test_input = np.array([78.3, -65.7, 102.85, -54.45, -8.85, 95.83, 77.4, 46.65, 118.34, -119.25, -77.4, 88.81, 76.95, -129.45, 85.67], dtype=np.float32)
# test_input = np.array([86.7,-65.55,90.51,-52.5,-8.7,70.91,75.15,53.1,96.07,-111.6,-93.6,87.6,90,-114.15,98.01], dtype=np.float32)
# test_input = test_input * denorm_vecB

gt_lst = []
pt_lst = []
for i in range(len(test_data_loader)):
    test_input, test_output = test_data_loader.dataset[i]
    test_input = test_input.numpy()
    test_output = test_output.numpy()
    # print("GT Input: ", test_input)
    print("GT Output: ", test_output)
    pytorch_output = infer_with_pytorch_model("reskin_model.pt", test_input)
    print("PT Output:", pytorch_output)
    gt_lst.append(test_output[2])
    pt_lst.append(pytorch_output[2                           ])

plt.plot(gt_lst, label='GT')
plt.plot(pt_lst, label='PT')
plt.pause(1)
plt.show()


# Perform inference with both models
pytorch_output = infer_with_pytorch_model("reskin_model.pt", test_input)
# tflite_output = infer_with_tflite_model("reskin_model.tflite", test_input)

# Print the results
print("PyTorch model prediction:", pytorch_output * denorm_vecF)
# print("TFLite model prediction:", tflite_output * denorm_vecF)
# print("PyTorch model prediction:", pytorch_output )
# print("TFLite model prediction:", tflite_output )