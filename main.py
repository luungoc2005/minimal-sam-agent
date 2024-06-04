from dotenv import load_dotenv
load_dotenv()

import os
import base64
import subprocess
import time
import re
import json

import cv2
from PIL import Image
import numpy as np
import openai

sam_checkpoint = os.getenv('SAM_CHECKPOINT_PATH')

# annotation contstants
screenshot_path = './screenshot.png'
annotated_path = './annotated.png'

mark_alpha = .5
font_scale = .4
thickness = 1
size = 512

# llm constants
prompt = (
    "You are an orchestrator controlling an Android phone.\n"
    "You need to perform the following task:\n{task_description}.\n\n"
    "Given the screenshot of the app, think about how what you should do to advance the task.\n"
    "The user will then reply with a screenshot of the app after the action is performed, or the result of the action.\n"
    "The screenshot will be annotated with ID numbers that you can use for actions.\n"
    "If you try something and it doesn't work, think if there's another way to do the task.\n"
    "When the task is finished, you have to use the [finish] action\n\n"
    "Available actions:\n"
    "- finish: Finish the task\n"
    "- get_otp: Get the OTP from SMS when you're asked\n"
    "- tap_at_location: Tap at a specific location on the screen. Example: tap_at_location 1\n"
    "- long_press_at_location: Tap and hold at a specific location on the screen. Example: long_press_at_location 1\n"
    "- enter_text_at_location: Enter text at a specific location on the screen. Example: enter_text_at_location 1 'Hello'\n"
    "- swipe_at_location: Swipe up/down/left/right at a specific location on the screen. Example: swipe_at_location 1 up\n"
    "- wait: Wait and do nothing for a few seconds. Use when something is in progress.\n"
    "You must reply in the following format:\n"
    "Thought: your rationale for the action\n"
    "Action: the action to be performed. Only use 1 action at a time\n"
)
MODEL = 'gpt-4o'
MAX_TOKENS = 100

# SAM

def get_mask_generator():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    model_type = "vit_b"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamAutomaticMaskGenerator(sam, 
        points_per_batch=64, 
        output_mode='uncompressed_rle'
    )

def load_image(image_path, transform_size=None):
    raw_image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = raw_image.size
    if transform_size is not None:
        from torchvision import transforms
        transform_fn = transforms.Compose([
            transforms.Resize(transform_size, interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        raw_image = transform_fn(raw_image)
    return raw_image, orig_width, orig_height

def compare_screenshots(prev_screenshot, curr_screenshot) -> float:
    # Compare two screenshots
    _prev_np = np.array(prev_screenshot)
    _curr_np = np.array(curr_screenshot)
    total = np.ones(_prev_np.shape) * 255
    result = (_curr_np - _prev_np).sum()
    return 1 - result / total.sum()

def write_text(img,
    text,
    text_offset_x=20,
    text_offset_y=20,
    vspace=8,
    hspace=5,
    font_scale=1.0,
    background_RGB=(228,225,222),
    text_RGB=(1,1,1),
    font=cv2.FONT_HERSHEY_DUPLEX,
    thickness=1,
    alpha=0.6,
    gamma=0
):
    try:
        R,G,B = background_RGB[:3]
        text_R,text_G,text_B = text_RGB[:3]
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
        x, y, w, h = text_offset_x, text_offset_y, text_width , text_height
        crop = img[y-vspace:y+h+vspace, x-hspace:x+w+hspace]
        white_rect = np.ones(crop.shape, dtype=np.uint8)
        b,g,r = cv2.split(white_rect)
        rect_changed = cv2.merge((B*b,G*g,R*r))

        res = cv2.addWeighted(crop, alpha, rect_changed, 1-alpha, gamma)
        img[y-vspace:y+vspace+h, x-hspace:x+w+hspace] = res

        cv2.putText(img, text, (x, (y+h)), font, fontScale=font_scale, color=(text_B,text_G,text_R), thickness=thickness, lineType=cv2.LINE_AA)
        return img
    except:
        return img

def contrasting_color(rgb):
    """Returns 'white' or 'black' depending on which color contrasts more with the given RGB value."""
    
    # Decompose the RGB tuple
    R, G, B = rgb

    # Calculate the Y value
    Y = 0.299 * R + 0.587 * G + 0.114 * B

    # If Y value is greater than 128, it's closer to white so return black. Otherwise, return white.
    return 'black' if Y > 128 else 'white'

def get_annotations(raw_image, anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    raw_image = np.array(raw_image)

    mark_coords = {}
    for ix, ann in enumerate(sorted_anns):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'{ix + 1}'
        box_coord = ann['bbox']
        x, y, w, h = box_coord
        # calculate text coords
        bbox_x, bbox_y = int(x), int(y)
        mark_coords[ix + 1] = int(bbox_x + w // 2), int(bbox_y + h // 2)

        bg_color_text = contrasting_color(raw_image[bbox_y, bbox_x][:3])
        bg_color = (0, 0, 0) if bg_color_text == 'black' else (255, 255, 255)
        text_color = (0, 0, 0) if bg_color_text == 'white' else (255, 255, 255)
        vspace = 8
        hspace = 5
        
        raw_image = write_text(
            raw_image, 
            text, 
            bbox_x + hspace, bbox_y + vspace, 
            vspace, hspace, font_scale, 
            bg_color,
            text_color,
            font, thickness, mark_alpha)
    return raw_image, mark_coords

if __name__ == '__main__':
    BASE_URL = os.getenv('BASE_URL')
    if BASE_URL is None:
        client = openai.Client()
    else:
        client = openai.Client(base_url=BASE_URL)

    mask_generator = get_mask_generator()
    with open('task.txt', 'r') as fp:
        agent_history = [
            {
                'role': 'system',
                'content': prompt.format(
                    task_description=fp.read(),
                )
            }
        ]

    def get_screenshot_reply():
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{base64_image}"

        subprocess.call(f"adb shell screencap -p > {screenshot_path}", shell=True)
        raw_image, orig_width, orig_height = load_image(screenshot_path, size)

        start_time = time.time()
        print('Generating masks...', end=' ')
        masks = mask_generator.generate(np.array(raw_image))
        end_time = time.time()
        print(f'Done in {end_time - start_time:.2f}s')

        raw_image, mark_coords = get_annotations(raw_image, masks)
        cv2.imwrite(annotated_path, cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR))
        _, new_width, new_height = load_image(annotated_path)
        screenshot_url = encode_image(annotated_path)
        # translate coords
        for key in mark_coords:
            x, y = mark_coords[key]
            # these x, y are for the new image size, must transform to the original size
            x = int(x * orig_width / new_width)
            y = int(y * orig_height / new_height)
            mark_coords[key] = x, y
        with open('mask_coords.json', 'w') as fp:
            json.dump({k: {'x': v[0], 'y': v[1]} for k, v in mark_coords.items()}, fp, indent=2)
        return {
            'role': 'user',
            'content': [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": screenshot_url,
                    },
                },
            ],
        }, mark_coords

    # handle the agent action and returns the reply
    def handle_action(action, mask_coords):
        action_params = action.split(' ')
        action_name = action_params[0].lower()

        # check if latest screenshot differs too much from previous
        orig_screenshot, _, _ = load_image(screenshot_path, size)
        subprocess.call(f"adb shell screencap -p > {screenshot_path}", shell=True)
        new_screenshot, _, _ = load_image(screenshot_path, size)
        similarity = compare_screenshots(orig_screenshot, new_screenshot)
        if similarity < .8:
            print('Screenshot changed too much. Trying again.')
            return get_screenshot_reply()

        if action_name == 'finish':
            return None, mask_coords
        elif action_name == 'tap_at_location':
            location = int(action_params[1])
            x, y = mask_coords[location]
            print('Tapping at', x, y)
            subprocess.Popen(f'adb shell input tap {x} {y}', shell=True)
            time.sleep(1)
            return get_screenshot_reply()
        elif action_name == 'long_press_at_location':
            location = int(action_params[1])
            x, y = mask_coords[location]
            print('Long pressing at', x, y)
            subprocess.Popen(f'adb shell input touchscreen swipe {x} {y} {x} {y} 3000', shell=True)
            time.sleep(1)
            return get_screenshot_reply()
        elif action_name == 'enter_text_at_location':
            location = int(action_params[1])
            text = ' '.join(action_params[2:])
            x, y = mask_coords[location]
            print('Tapping at', x, y)
            subprocess.Popen(f'adb shell input tap {x} {y}', shell=True)
            time.sleep(1)
            print('Typing', text)
            subprocess.Popen(f"adb shell input text $(echo '{text}' | sed 's/ /\%s/g')", shell=True)
            time.sleep(1)
            return get_screenshot_reply()
        elif action_name == 'swipe_at_location':
            location = int(action_params[1])
            direction = action_params[2].lower()
            x, y = mask_coords[location]
            swipe_distance = 600
            if direction == 'up':
                subprocess.Popen(f'adb shell input swipe {x} {y} {x} {y - swipe_distance}', shell=True)
            elif direction == 'down':
                subprocess.Popen(f'adb shell input swipe {x} {y} {x} {y + swipe_distance}', shell=True)
            elif direction == 'left':
                subprocess.Popen(f'adb shell input swipe {x} {y} {x - swipe_distance} {y}', shell=True)
            elif direction == 'right':
                subprocess.Popen(f'adb shell input swipe {x} {y} {x + swipe_distance} {y}', shell=True)
            time.sleep(1)
            return get_screenshot_reply()
        elif action_name == 'wait':
            time.sleep(5)
            return get_screenshot_reply()
        else:
            return {
                'role': 'user',
                'content': 'Invalid action. Try again'
            }, mask_coords

    def get_response(client: openai.Client, history, max_tokens=MAX_TOKENS):
        # clear previous screenshots for cost saving
        prompt_history = []
        for ix, item in enumerate(history):
            if not isinstance(item['content'], str) and ix != len(history) - 1:
                prompt_history.append({
                    'role': item['role'],
                    'content': '(screenshot)'
                })
            else:
                prompt_history.append(item)

        start_time = time.time()
        print('Getting response...', end=' ')
        response = client.chat.completions.create(
            model=MODEL,
            messages=prompt_history,
            max_tokens=max_tokens,
            temperature=.2,
        )
        end_time = time.time()
        print(f'Done in {end_time - start_time:.2f}s')
        new_message = response.choices[0].message
        history.append(new_message.to_dict())
        return history, new_message

    # main loop
    time.sleep(3) # wait for app start
    user_reply, mask_coords = get_screenshot_reply()
    while user_reply is not None:
        agent_history.append(user_reply)
        agent_history, agent_reply = get_response(client, agent_history)
        print(agent_reply.content)
        with open('./history.json', 'w') as fp:
            json.dump(agent_history, fp, indent=2)
        regex = (
            r"Action\s*:(.*?)$"
        )
        action_match = re.search(regex, agent_reply.content, re.MULTILINE | re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            user_reply, mask_coords = handle_action(action, mask_coords)
        else:
            user_reply = None
        print('===')
    