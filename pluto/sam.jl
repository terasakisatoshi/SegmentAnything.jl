### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 1a9f06b0-ebf9-11ee-2e9d-bb6f116fff54
begin
	using Pkg
	Pkg.activate(;temp=true)
	Pkg.add(
		PackageSpec(
			url="https://github.com/terasakisatoshi/SegmentAnything.jl", rev="terasaki/pluto-notebook"
		)
	)
	Pkg.add(["Images", "ImageShow", "ImageDraw", "PythonCall"])
	Pkg.add(["PlutoUI"])

	using SegmentAnything: ImageMask, SamPredictor
	using SegmentAnything: SamAutomaticMaskGenerator, generate
end

# ╔═╡ 5e1544cc-a143-4769-8827-21d960d49167
begin
	using Downloads: download
	
	using Images: load, RGB, Gray, N0f8, colorview
	using ImageShow
	using ImageDraw: ImageDraw, Polygon, Cross, Point, RectanglePoints, draw, draw!
	using PythonCall: pyconvert
end

# ╔═╡ 167d404c-0e43-4dd0-aff4-c30e2d36d290
using PlutoUI

# ╔═╡ e087ee47-d483-44cc-a152-6ee517ff1793
md"""
# SegmentAnything

This notebook demonstrates [segment-anything](https://github.com/facebookresearch/segment-anything) in Julia. There is a Julia package named [rafaqz/SegmentAnything.jl](https://github.com/rafaqz/SegmentAnything.jl) that provides a thin wrapper for SAM(segment anything model). 

There is a problem of loading the original SegmentAnything.jl. To avoid the installation problem, we have forked the original repository and released a repository with some modifications.
"""

# ╔═╡ 97b34600-a088-4187-b1d2-7925777811bc
md"""
## Setup SegmentAnything.jl
"""

# ╔═╡ 3aa93c57-e4eb-4ead-85d4-900a3728a2aa
md"""
## Load Julia packages
"""

# ╔═╡ 460d3514-7297-4444-8358-94f0f6986142
md"""
## Prepare a test image

As shown in the README of [rafaqz/SegmentAnything.jl](https://github.com/rafaqz/SegmentAnything.jl), we will use the same image as a test image. There is a kitty and beagle in the image. In the next section, we would like to segment them.
"""

# ╔═╡ 7b7757d9-f3f0-47f0-aa21-51ba38b62de8
begin
	url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Beagle_and_sleeping_black_and_white_kitty-01.jpg"
	download(url, "pic.jpg")
	img = load("pic.jpg")
end

# ╔═╡ 5876c2b7-e5f8-40ff-b379-7724420d5791
md"""
## Segment an image with `point_coords` information

By specifying the areas of interest to the user, SAM generates a mask that extracts those areas. For the test image as introduced the previous section above, it is easy to observe the red cross `(x, y) = (400, 800)` specifies the kitty and the green cross `(x, y) = (400, 600)` specifies the beagle respectively.
"""

# ╔═╡ 1e6f525f-9127-4ef2-be38-0ed922adf1ca
let
	_img = copy(img)
	# draw a cross for kitty
	draw!(_img, Cross(Point(400, 800), 50), RGB{N0f8}(1,0,0))
	# draw a cross for beagle
	draw!(_img, Cross(Point(400, 600), 50), RGB{N0f8}(0,1,0))
end

# ╔═╡ ffc8e0a1-0113-4071-87de-6cb03c4ba843
md"""
`SegmentAnything.jl` provides a predictor type named `SamPredictor`. It accepts `point_coords` that represents the point of interest to the user.

```julia
SamPredictor(; checkpoint=DEFAULT_MODEL, device="cuda")
```

Node that to work on a machine without a GPU accelerator, we must set keyword argument `device` to `"cpu"`.
"""

# ╔═╡ 14515040-ee77-45a4-a531-bef5902898af
predictor = SamPredictor(device="cpu")

# ╔═╡ e93b02ac-b010-440a-ac05-74980caaf681
md"""
Let's extract the kitty from the test image. The `ImageMask` type is useful for doing just that.
"""

# ╔═╡ 8100b3ed-d1fd-44db-9ca3-715633f719a1
begin
	kitty_point = (400, 800)
	mask1 = ImageMask(predictor, img; 
	  point_coords=[kitty_point],
	  point_labels=[true],
	)
end

# ╔═╡ 47a5a45d-179e-40fa-8c3d-9059f3083022
md"""
The `mask1` is a type of `ImageMask` and has the following fields:
"""

# ╔═╡ 7f00e24e-a5aa-46cb-a85e-1487bb13d765
fieldnames(typeof(mask1))

# ╔═╡ 3180c50d-e08c-4d26-bdbd-4937dfe0aea7
md"""
We can get the mask `mask1.masks[2, :, :]` that covers the kitty which is what we wanted.
"""

# ╔═╡ 13e0947e-d5ae-41d6-bb48-7cdd5900c90b
let
	img_with_point = draw(img, Cross(Point(kitty_point), 50), RGB{N0f8}(1,0,0))
	mask = colorview(Gray, mask1.masks[2, :, :])
	[img_with_point mask]
end

# ╔═╡ 888da19f-1876-4c42-9bcf-6ffe437a60a1
md"""
Below is the result showing the kitty is extracted from the test image.
"""

# ╔═╡ 4c64557b-a731-4526-a608-68ad3c48bffa
let
	kitty = copy(img)
	mask = mask1.masks[2, :, :]
	for i in eachindex(kitty, mask)
		kitty[i] = kitty[i] * mask[i]
	end
	kitty
end

# ╔═╡ 46d7e4ab-2be0-4266-a3ab-e41265997f98
md"""
Similarly, let's extract the beagle from the test image.
"""

# ╔═╡ f259ad3e-d8fe-4873-ae20-4fdd961d6fec
begin
	beagle_point = (400, 600)
	mask2 = ImageMask(predictor, img; 
	  point_coords=[beagle_point],
	  point_labels=[true],
	);
end

# ╔═╡ aef1f017-9371-403f-9882-771c9ed42185
let
	img_with_point = draw(img, Cross(Point(beagle_point), 50), RGB{N0f8}(1,0,0))
	mask = colorview(Gray, mask2.masks[2, :, :])
	[img_with_point mask]
end

# ╔═╡ ae857d67-e2a0-454e-b789-8874a8373847
md"""
## Using `SamAutomaticMaskGenerator`

You can generate masks for an entire image without specifying the point of interest using SamAutomaticMaskGenerator.
"""

# ╔═╡ 402634fa-61ec-483e-a091-0d6183eca490
begin
	generator = SamAutomaticMaskGenerator()
	out = generate(generator, img)
end

# ╔═╡ 877d4029-70ae-4dbc-bd28-3530284cadb9
md"""
The `out` an array of an objects. Each object stores information about `segmentation`(mask), `bbox`(bounding box for the object).
"""

# ╔═╡ fe915ccc-ece8-40e4-9a54-679b31e23fad
@bind maskid Select(collect(Base.OneTo(length(out))))

# ╔═╡ a1207043-3d8c-45d5-96a4-7a1b04a0db88
let
	x, y, w, h = pyconvert(Array, out[maskid]["bbox"])
	x1 = x
	y1 = y
	x2 = x + w
	y2 = y + h
	point_coords = first(out[maskid]["point_coords"])
	mask = colorview(Gray, pyconvert(Array, out[maskid]["segmentation"]))
	_img = copy(img)
	draw!(
		mask, 
		Polygon(RectanglePoints(CartesianIndex(x1, y1), CartesianIndex(x2, y2))), Gray{Bool}(1)
	)
	img_with_bbox = draw(
		img, 
		Polygon(RectanglePoints(CartesianIndex(x1, y1), CartesianIndex(x2, y2))), RGB{N0f8}(1, 0, 0)
	)
	[img_with_bbox mask]
end

# ╔═╡ Cell order:
# ╟─e087ee47-d483-44cc-a152-6ee517ff1793
# ╟─97b34600-a088-4187-b1d2-7925777811bc
# ╠═1a9f06b0-ebf9-11ee-2e9d-bb6f116fff54
# ╟─3aa93c57-e4eb-4ead-85d4-900a3728a2aa
# ╠═5e1544cc-a143-4769-8827-21d960d49167
# ╠═167d404c-0e43-4dd0-aff4-c30e2d36d290
# ╟─460d3514-7297-4444-8358-94f0f6986142
# ╠═7b7757d9-f3f0-47f0-aa21-51ba38b62de8
# ╟─5876c2b7-e5f8-40ff-b379-7724420d5791
# ╠═1e6f525f-9127-4ef2-be38-0ed922adf1ca
# ╟─ffc8e0a1-0113-4071-87de-6cb03c4ba843
# ╠═14515040-ee77-45a4-a531-bef5902898af
# ╟─e93b02ac-b010-440a-ac05-74980caaf681
# ╠═8100b3ed-d1fd-44db-9ca3-715633f719a1
# ╟─47a5a45d-179e-40fa-8c3d-9059f3083022
# ╠═7f00e24e-a5aa-46cb-a85e-1487bb13d765
# ╟─3180c50d-e08c-4d26-bdbd-4937dfe0aea7
# ╠═13e0947e-d5ae-41d6-bb48-7cdd5900c90b
# ╟─888da19f-1876-4c42-9bcf-6ffe437a60a1
# ╠═4c64557b-a731-4526-a608-68ad3c48bffa
# ╟─46d7e4ab-2be0-4266-a3ab-e41265997f98
# ╠═f259ad3e-d8fe-4873-ae20-4fdd961d6fec
# ╠═aef1f017-9371-403f-9882-771c9ed42185
# ╟─ae857d67-e2a0-454e-b789-8874a8373847
# ╠═402634fa-61ec-483e-a091-0d6183eca490
# ╟─877d4029-70ae-4dbc-bd28-3530284cadb9
# ╠═fe915ccc-ece8-40e4-9a54-679b31e23fad
# ╠═a1207043-3d8c-45d5-96a4-7a1b04a0db88
