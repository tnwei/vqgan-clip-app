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

## Multi-stage iteration

The output image of the previous run can be carried over as the input image of the next run in the VQGAN-CLIP webapp, using the `continue_previous_run` option. This allows you to continue iterating on an image if you like how it has turned out thus far. Extending upon that feature enables multi-stage iteration, where the same image can be iterated upon using different prompts at different stages.

For example, you can tell the network to generate "Cowboy singing in the sky", then continue the same image and weights using a different prompt, "Fish on an alien planet under the night sky". Because of how backprop works, the network will find the easiest way to change the previous image to fit the new prompt. Should be useful for preserving visual structure between images, and for smoothly transitioning from one scene to another.

Here is an example where "Backyard in spring" is first generated, then iterated upon with prompts "Backyard in summer", "Backyard in autumn", and "Backyard in winter". Major visual elements in the initial image were inherited and utilized across multiple runs.

![Backyard in spring, summer, autumn and winter](docs/images/four-seasons-20210808.jpg)

A few things to take note:
+ If a new image size is specified, the existing output image will be cropped to size accordingly.
+ This is specifically possible for VQGAN-CLIP but not for CLIP guided diffusion. (Explain how both of them work)
+ Splitting a long run into multiple successive runs using the same prompt do not yield the same outcome due to the underlying stochasticity. This randomness can't be mitigated by setting the random seed alone. See the section on reproducibility in notes-and-observations.md. 
