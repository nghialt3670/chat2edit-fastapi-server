VI_PROMPT_TEMPLATE = """
Only use these methods:

{methods}

Example 1:
<observation>
    user_message(text="Hãy xóa con mèo đi", images=[image])
</observation>
<thinking>
    We need to detect the cat first.
</thinking>
<action>
    cat = detect(image, "cat")
</action>
<observation>
    sys_warning("Detected 2 'cat' in image")
</observation>
<thinking>
    There are 2 detected cats in the image, we need to ask user to specify which one.
</thinking>
<action>
    response(text="Trong ảnh có 2 con mèo, bạn muốn xóa con nào?")
<action>

Example 2:
<observation>
    user_message(text="Hãy tăng độ sáng của bức ảnh", images=[image0])
</observation>
<thinking>
    We need to ask user how much brightness they want to increase.
</thinking>
<action>
    response(text="Bạn muốn tăng độ sáng bao nhiêu")
</action>
<observation>
    user_message(text="Khoảng 20%")
</observation>
<thinking>
    We have all the information we need, lets edit the image.
</thinking>
<action>
    brightness_filter = create_filter("brightness", 1.2)
    brighten_image = apply_filter(image, brightness_filter)
    response(text="Đây là ảnh sau khi đã tăng độ sáng 20%", images=[brighten_image])
</action>

Give your next thinking and action base on the observation:
"""
