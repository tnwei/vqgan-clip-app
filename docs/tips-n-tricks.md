# Tips and tricks

## Prompt weighting

The text prompts allow separating text prompts using the "|" symbol, with custom weightage for each. Example: `A beautiful sunny countryside scene of a Collie dog running towards a herd of sheep in a grassy field with a farmhouse in the background:100 | wild flowers:30 | roses:-30 | photorealistic:20 | V-ray:20 | ray tracing:20 | unreal engine:20`. 

Refer to [this Reddit post](https://www.reddit.com/r/bigsleep/comments/p15fis/tutorial_an_introduction_for_newbies_to_using_the/) for more info. 

## Using long, descriptive prompts

CLIP can handle long, descriptive prompts, as long as the prompt is understandable and not too specific. Example: [this Reddit post](https://www.reddit.com/r/MediaSynthesis/comments/oej9qc/gptneo_vqganclip/)

## Prompt engineering for stylization

Art style of the images generated can be somewhat controlled via prompt engineering. 

+ Adding mentions of "Unreal engine" in the prompt causes the generated output to resemble a 3D render in high resolution. Example: [this Tweet](https://twitter.com/arankomatsuzaki/status/1399471244760649729?s=20)
+ The art style of specific artists can be mimicked by adding their name to the prompt. [This blogpost](https://moultano.wordpress.com/2021/07/20/tour-of-the-sacred-library/) shows a number of images generated using the style of James Gurney. [This imgur album](https://imgur.com/a/Ha7lsYu) compares the output of similar prompts with different artist names appended.

