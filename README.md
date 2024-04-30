# Freecell image generator
Generates heavily distorted / augmented images of Freecell solitaire games for use training a Yolov4 model to recognise them
## Requirements
### Python packages for:
Numpy

Albumentations

OpenCV

### Images to use to build the composites:
Card images in cards/[0-51].png

Alternatively these can be generated using an image like https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/English_pattern_playing_cards_deck.svg/1280px-English_pattern_playing_cards_deck.svg.png using the function split_img_into_cards, which will use opencv to split out cards with a green background (alter the hue used in mask_background creation for a different background colour)

The describable textures dataset (https://www.robots.ox.ac.uk/~vgg/data/dtd/), in a folder named "dtd", for backgrounds
