from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate

# 定义输入层
input_shape = (224, 224, 3)
input_layer = Input(shape=input_shape)

# 定义卷积层和池化层
conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flatten = Flatten()(pool2)

# 定义用于预测类别的输出层
class_output = Dense(3, activation='softmax', name='class_output')(flatten)  # 假设有3个类别

# 定义用于预测长宽比的输出层
aspect_ratio_output = Dense(1, activation='linear', name='aspect_ratio_output')(flatten)

# 定义模型
model = Model(inputs=input_layer, outputs=[class_output, aspect_ratio_output])

# 编译模型
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy', 'aspect_ratio_output': 'mean_squared_error'},
              metrics={'class_output': 'accuracy', 'aspect_ratio_output': 'mae'})


from keras import backend as K

# 自定义损失函数
def custom_loss(y_true, y_pred):
    class_true, aspect_ratio_true = y_true
    class_pred, aspect_ratio_pred = y_pred

    # 计算类别损失
    class_loss = K.categorical_crossentropy(class_true, class_pred)

    # 计算长宽比损失
    aspect_ratio_loss = K.square(aspect_ratio_true - aspect_ratio_pred)

    # 将类别损失和长宽比损失结合起来
    total_loss = class_loss + aspect_ratio_loss

    return total_loss

# 编译模型
model.compile(optimizer='adam',
              loss=custom_loss,
              metrics=['accuracy'])


from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate

# 定义输入层
input_shape = (224, 224, 3)
input_layer = Input(shape=input_shape)

# 定义卷积层和池化层
conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flatten = Flatten()(pool2)

# 定义用于预测类别的输出层
class_output = Dense(2, activation='softmax', name='class_output')(flatten)  # 假设有2个类别：正常、转移

# 定义用于预测回声特征的输出层
echo_feature_output = Dense(1, activation='linear', name='echo_feature_output')(flatten)

# 定义模型
model = Model(inputs=input_layer, outputs=[class_output, echo_feature_output])

# 编译模型
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy', 'echo_feature_output': 'mean_squared_error'},
              metrics={'class_output': 'accuracy', 'echo_feature_output': 'mae'})


from keras import backend as K

# 自定义损失函数
def custom_loss(y_true, y_pred):
    class_true, echo_feature_true = y_true
    class_pred, echo_feature_pred = y_pred

    # 计算类别损失
    class_loss = K.categorical_crossentropy(class_true, class_pred)

    # 计算回声特征损失
    echo_feature_loss = K.square(echo_feature_true - echo_feature_pred)

    # 添加对淋巴结中心回声特征的惩罚项
    threshold = 0.5
    if echo_feature_true < threshold and class_true == '转移':
        echo_feature_loss += 0.5  # 举例：增加对淋巴结中心呈单一的低回声或弱回声为转移的惩罚

    # 将类别损失和回声特征损失结合起来
    total_loss = class_loss + echo_feature_loss

    return total_loss
