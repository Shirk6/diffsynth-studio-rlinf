from ..models.wan_video_dit import WanModel
from ..models.wan_video_vae import WanVideoVAE, WanVideoVAE38

model_loader_configs = [
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    # 新加的
    
    (None, "3803bd491e969952cbbeb1a48fd8fa11", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "d9c91561d7feb8510857001e88c410db", ["wan_video_dit"], [WanModel], "civitai"), # 14B I2V 
    (None, "16266d0dbcda81a4f797d7b8aafe0f35", ["wan_video_dit"], [WanModel], "civitai"), # 14B I2V control context lora
    (None, "dd9ee3cc97642977beab8e12d2610a19", ["wan_video_dit"], [WanModel], "civitai"), # 14B I2V control context lora 最新
    (None, "0c81d47223c50d8d74106f79310ae207", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    (None, "3f4e37438f72ef88cd27b161fd1b193c", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    (None, "d3abb829857dff2d9129d2f396a7eace", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    (None, "bc4824aef7c3f23d3378cec6e2b1316c", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    (None, "fcc43a93949201bafeb34aa1eb8bc50f", ["wan_video_dit"], [WanModel], "civitai"), # 5B TA2V action (agx/action_dim=10)
    (None, "6efc7e19e87c1755f17dec14be3d0bf1", ["wan_video_dit"], [WanModel], "civitai"), # 5B TA2V action
    #
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"), # 14B I2V
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"), #5B TI2V
    (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5ec04e02b42d2580483ad69f4e76346a", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),

    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),

    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
]