import math
import cv2


def rotate_image(mat, angle, point):
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (width / 2,
                    height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    # image_center = point

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def get_angle_rotation(centre, point, target_angle):
    # centre - Точка относительно которой надо вращать. tuple (x,y)
    # point - точка которую надо повернуть tuple (x,y)
    # угол куда должна повернуться point. градусы от 0 до 360.   Принцип такой: 15 часов - 0 градусов, 12 часов - 90 градусов,   9 часов - 180 градусов,  18 часов - 270 грудусов.  Отрицательных градусов быть не должно

    new_point = (point[0] - centre[0], point[1] - centre[
        1])  # передвигаем центр системы координат в точку центр. у нее будет (0,0) ищем новые координаты у точки point
    a, b = new_point[0], new_point[1]
    res = math.atan2(b, a)  # ищем полярный угол у new_point
    if res < 0:
        res += 2 * math.pi
    return (math.degrees(res) + target_angle) % 360  # возвращаем угол поворота для cv2


def get_image_after_rotation(img, model):
    results = model(img)
    pd = results.pandas().xyxy[0]
    pd = pd.assign(centre_x=pd.xmin + (pd.xmax - pd.xmin) / 2)
    pd = pd.assign(centre_y=pd.ymin + (pd.ymax - pd.ymin) / 2)

    tmp = pd.loc[pd['name'] == 'svidetelstvo']

    N, V = None, None
    for index, row in tmp.iterrows():
        N = (row['centre_x'], row['centre_y'])
        break
    # получим координаты верха, там где печать
    tmp = pd.loc[pd['name'] == 'ts']
    for index, row in tmp.iterrows():
        V = (row['centre_x'], row['centre_y'])
        break
    if N == None or V == None:  # похоже там нет нужных нам строк
        return img

    angle = get_angle_rotation(N, V, 0)
    return rotate_image(img, angle, N)


def get_rotated(img, model):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = get_image_after_rotation(image, model)
    image = get_image_after_rotation(image, model)  # второй подряд поворот еще лучше выравнивает.
    return image
