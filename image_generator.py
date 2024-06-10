import cv2
import numpy as np
import random
from enum import Enum
import albumentations as a
import os
import math
import pathlib
import typing
from typing import Tuple, List

def show(to_show):
    cv2.imshow('image', to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_multiple(*to_show):
    for i, im in enumerate(to_show):
        cv2.imshow(f'image_{i}', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mask_to_image(mask):
    return cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

def split_img_into_cards(deck):
    as_hsv = cv2.cvtColor(deck, cv2.COLOR_BGR2HSV)
    mask_background = (as_hsv[:, :, 0] == 60).astype("uint8")
    mask_foreground = 1 - mask_background

    transparent = cv2.cvtColor(deck, cv2.COLOR_BGR2BGRA)
    foreground_only = cv2.bitwise_and(transparent, transparent, mask=mask_foreground)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_foreground)
    j = 0
    for i in range(1, num_labels):
        area = stats[i][cv2.CC_STAT_AREA]
        if area < 1000:
            continue;
        left = stats[i][cv2.CC_STAT_LEFT]
        top = stats[i][cv2.CC_STAT_TOP]
        width = stats[i][cv2.CC_STAT_WIDTH]
        height = stats[i][cv2.CC_STAT_HEIGHT]
        right = left + width
        bottom = top + height
        subimage = foreground_only[top:bottom, left:right]
        cv2.imwrite('cards/{}.png'.format(j), subimage)
        j += 1
    if j != 52:
        print(f"ERROR: Wrong number of cards generated ({j})")

class Suit(Enum):
    SPADES = "spades"
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"

SUIT_ORDER = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]


def generate_obj_names():
    i = 0
    for suit in SUIT_ORDER:
        print(f"ace of {suit.value}")
        for card_num in range(2, 11):
            print(f"{card_num} of {suit.value}")
        print(f"jack of {suit.value}")
        print(f"queen of {suit.value}")
        print(f"king of {suit.value}")

def split_file(filename):
    deck = cv2.imread(filename)
    split_img_into_cards(deck)


def build_card_transform():
    return a.Compose([
        a.Rotate(limit=10, p=1),
        a.Rotate(limit=30, p=0.10),
        a.RandomGamma(),
        ], bbox_params=a.BboxParams(format='coco', label_fields=["class_labels"]))

def build_game_transform():
    return a.Compose([
        a.Perspective(scale=0.10),
        a.PixelDropout(),
        a.GaussNoise(),
        a.RandomGamma(),
        a.RandomShadow(),
        a.RandomToneCurve(),
        ], bbox_params=a.BboxParams(format='coco', label_fields=["class_labels"]))

CARD_TEXTS=["A"] + list(map(lambda x: str(x), list(range(2,11)))) + ["J", "Q", "K"]
def class_to_string(class_number):
    suit_number = class_number // 13
    card_number = class_number % 13 + 1
    card_text = card_texts[card_number]
    suit = SUIT_ORDER[suit_number]
    return f"{card_text}{suit.value[0]}"

def add_border_to_image(image, border_size):
    shape = list(image.shape)
    shape[0] += border_size * 2
    shape[1] += border_size * 2
    ret = np.zeros(shape, "uint8")
    ret[border_size:border_size+image.shape[0], border_size:border_size+image.shape[1]] = image
    return ret


def draw_bounding_boxes_on_image(image, bounding_boxes, labels = None):
    for (i, [x, y,width,height]) in enumerate(bounding_boxes):
        cv2.rectangle(image, (int(x),int(y)), (int(x+width), int(y + height)), (0, 255, 0))
        if labels is not None:
            cv2.putText(image, text=class_to_string(labels[i]), org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=cv2.COLORMAP_INFERNO)

Image = typing.Any
BoundingBox = Tuple[int, int, int, int]
Deck = Tuple[List[BoundingBox], List[Image]]
def read_bounding_box_file(filename):
    with open(filename) as bounding_boxes_file:
        ret = []
        while True:
            next_line: str = bounding_boxes_file.readline()
            if not next_line:
                break

            # Skip empty lines
            trimmed = next_line.strip()
            if not trimmed:
                continue


            if trimmed[0] != '[' or trimmed[-1] != ']':
                raise Exception(f"Expected nonempty lines in {filename} to start with '[' and end with ']'")

            bounding_boxes = list(map(lambda s: int(s), map(lambda x: x.strip(), trimmed[1:-1].split(","))))
            if len(bounding_boxes) != 4:
                raise Exception(f"Expected nonempty lines in {filename} to contain 4 comma seperated integer values, found {len(bounding_boxes)} values")

            ret.append(bounding_boxes)

        if len(ret) == 1:
            ret = [ret[0] for i in range(0, 52)]
        if len(ret) != 52:
            raise Exception(f"Expected 1 or 52 bounding boxes in {filename}, got {len(ret)}")
        return ret


def get_decks() -> List[Deck]:
    "Get card images and bounding box data, leaving alpha channel intact."
    decks = os.listdir("cards")
    ret = []
    for deck_folder_name in decks:
        bounding_boxes_file_name = f"cards/{deck_folder_name}/bounding_boxes.txt"
        bounding_boxes = read_bounding_box_file(bounding_boxes_file_name)
        cards = []
        for i in range(0, 52):
            image = cv2.imread(f"cards/{deck_folder_name}/{i}.png", cv2.IMREAD_UNCHANGED)
            cards.append(image)
        ret.append((bounding_boxes, cards))
    return ret;

def overlay_card(base_img, card_image, x, y):
    "Overlays a card, skipping pixels that are fully transparent. Partial transparency is ignored. x any y coordinates of top left of card dest"
    card_height = card_image.shape[0]
    card_width = card_image.shape[1]
    card_channels = card_image.shape[2]
    cutout=base_img[y:y+card_height, x:x+card_width, :]
    card_without_transparency = card_image[:,:,0:3]
    non_transparent = (card_image[:,:,3] != 0).astype("uint8")
    for channel in range(0,3):
        cutout[non_transparent == 1] = 0
    cv2.add(cutout, card_without_transparency, cutout, mask=non_transparent)
    base_img[y:y+card_height, x:x+card_width, :] = cutout

def get_card_bounding_boxes(x,y):
       return [[x + 2 , y + 3, 5, 17]] #  top left x y, width, height. ignoring bottom left for now

def translate_bounding_box(bb, x, y):
    ret = list(bb)
    ret[0] += x
    ret[1] += y
    return ret

def translate_bounding_boxes(bbs, x, y):
    return map(lambda bb: translate_bounding_box(bb, x, y), bbs)

def get_random_background():
    subdirs = os.listdir("dtd/images")
    subdir = subdirs[random.randrange(0, len(subdirs))]
    images = os.listdir(f"dtd/images/{subdir}")
    image = images[random.randrange(0, len(images))]
    return cv2.imread(f"dtd/images/{subdir}/{image}")


COLUMN_OFFSET=30
COLUMN_RANDOM_OFFSET=50
CARD_VERTICAL_OFFSET_PROPORTION = 0.2
CARD_RANDOM_VERTICAL_OFFSET_PROPORTION = 0.2
CARD_HORIZONTAL_JITTER = 10
NUM_CARDS_IN_LONG_COLUMN = 7
NUM_CARDS_IN_SHORT_COLUMN = 6
NUM_COLUMNS = 8
NUM_LONG_COLUMNS = 4
BONUS_BORDER_SPACE = 0
def get_output_image_dimensions(card_width: int, card_height: int) -> tuple[int, int, int]:
    max_vertical_offset_per_card = (CARD_RANDOM_VERTICAL_OFFSET_PROPORTION + CARD_VERTICAL_OFFSET_PROPORTION) * card_height
    extra_space_for_transforms = max(card_width, card_height) * 2
    max_column_height = card_height + max_vertical_offset_per_card * NUM_CARDS_IN_LONG_COLUMN
    max_column_width = COLUMN_OFFSET + COLUMN_RANDOM_OFFSET + card_width
    border_space = BONUS_BORDER_SPACE + extra_space_for_transforms
    total_vertical_space = border_space * 2 + max_column_height
    total_horizontal_space = border_space * 2 + max_column_width * NUM_COLUMNS
    return (int(border_space), int(total_horizontal_space), int(total_vertical_space))

def tile_background_to_size(background, width_needed, height_needed):
    background_vertical_repeat = max(1, math.ceil(height_needed / background.shape[0]))
    background_horizontal_repeat = max(1, math.ceil(width_needed / background.shape[1]))
    ret_image = cv2.repeat(background, background_horizontal_repeat, background_horizontal_repeat)
    ret_image = ret_image[0:height_needed, 0:width_needed,:]
    return ret_image

def get_next_column_x(this_column_x: int, card_width: int) -> int:
    return this_column_x + card_width + COLUMN_OFFSET + random.randrange(0, COLUMN_RANDOM_OFFSET)

def get_next_card_y(this_card_y: int, card_height: int) -> int:
    return (this_card_y
            + CARD_VERTICAL_OFFSET_PROPORTION * card_height
            + random.randrange(0, int(CARD_RANDOM_VERTICAL_OFFSET_PROPORTION * card_height)))

def generate_freecell_game(deck: Deck):
    bounding_boxes = deck[0]
    card_images = deck[1]
    card_width = card_images[0].shape[1]
    card_height = card_images[0].shape[0]
    (border_space, width_needed, height_needed) = get_output_image_dimensions(card_width, card_height)
    print(f"Generating game with dimensions {width_needed}, {height_needed}")
    background = None
    while background is None:
        background = get_random_background()

    ret_image = tile_background_to_size(background, width_needed, height_needed)

    ret_bounding_boxes = []
    ret_labels = []
    cards = list(range(0,52))
    random.shuffle(cards)
    column_x = border_space
    card_transform = build_card_transform()
    for column in range(0, NUM_COLUMNS):
        column_x = get_next_column_x(column_x, card_width)
        if column < NUM_LONG_COLUMNS:
            cards_needed = NUM_CARDS_IN_LONG_COLUMN
        else:
            cards_needed = NUM_CARDS_IN_SHORT_COLUMN
        card_y = border_space
        for card_number in range(0, cards_needed):
            card_x = column_x + random.randrange(-CARD_HORIZONTAL_JITTER, CARD_HORIZONTAL_JITTER)
            card_y = get_next_card_y(card_y, card_height)
            next_card = cards.pop()
            card_image = card_images[next_card]
            card_image_with_border = add_border_to_image(card_image, card_image.shape[0])
            card_bounding_boxes = [translate_bounding_box(bounding_boxes[next_card], card_image.shape[0], card_image.shape[0])]
            card_transformed = card_transform(image=card_image_with_border, bboxes=card_bounding_boxes, class_labels=[next_card])
            transformed_bounding_boxes = card_transformed["bboxes"]
            draw_card_x = int(card_x - card_image.shape[1])
            draw_card_y = int(card_y - card_image.shape[0])
            translated_boxes = list(translate_bounding_boxes(transformed_bounding_boxes, draw_card_x, draw_card_y))
            if len(translated_boxes) != 1:
                raise Exception("After transformation card had no bounding boxes!")
            ret_bounding_boxes.extend(translated_boxes)
            overlay_card(ret_image, card_transformed["image"], draw_card_x, draw_card_y)
            ret_labels.append(next_card)
    transform = build_game_transform()
    return transform(image=ret_image, bboxes=ret_bounding_boxes, class_labels=ret_labels)

def write_object_data(file, bboxes, class_labels):
    "Write data in the format <object-class> <x_center> <y_center> <width> <height>"
    if len(bboxes) != len(class_labels):
        raise Exception(f"Mismatch: {len(bboxes)} bboxes vs {len(class_labels)} class_labels")
    for (bbox, class_label) in zip(bboxes, class_labels):
        file.write(f"{class_label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def generate_many_games(num_to_generate,location="games", generate_images_with_boxes=False):
    decks = get_decks()
    for i in range(0, num_to_generate):
        if i % 100 == 0:
            print(f"Processing {i} of {num_to_generate}")
        game = generate_freecell_game(decks[random.randrange(0, len(decks))])
        game_image = game["image"]
        bboxes_coco = game["bboxes"]
        bboxes_albumentations = a.core.bbox_utils.convert_bboxes_to_albumentations(bboxes_coco, "coco", game_image.shape[0], game_image.shape[1])
        bboxes_yolo = a.core.bbox_utils.convert_bboxes_from_albumentations(bboxes_albumentations, "yolo", game_image.shape[0], game_image.shape[1])
        cv2.imwrite(f"{location}/{str(i)}.png", game_image)
        if generate_images_with_boxes:
            draw_bounding_boxes_on_image(game_image, bboxes_coco)
            cv2.imwrite(f"{location}/{str(i)}_with_boxes.png", game_image)
        with open(f"{location}/{str(i)}.txt", "w") as file:
            write_object_data(file, bboxes_yolo, game["class_labels"])


def print_train_txt(image_count):
    for i in range(0, image_count):
        print(f"data/obj/{i}.png")

if __name__ == "__main__":
    generate_many_games(10)
