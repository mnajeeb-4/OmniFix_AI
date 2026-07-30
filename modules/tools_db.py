"""
Tool database for OmniFix AI.
Contains structured information for each mechanical tool with icons, usage guides, and safety tips.
"""

TOOLS_DB = {
    "Wrenches": {
        "name": "Combination Wrench Set",
        "description": "Used for tightening/loosening nuts and bolts.",
        "usage": "Select the correct size, place on the fastener, and turn counter‑clockwise to loosen, clockwise to tighten.",
        "icon_url": "https://img.icons8.com/color/96/wrench.png",
        "safety_tips": "Always use the correct size to avoid rounding the fastener; wear gloves to prevent injury."
    },
    "Multimeters": {
        "name": "Digital Multimeter",
        "description": "Measures voltage, current, resistance, and continuity.",
        "usage": "Set the dial to the appropriate range, connect probes to the circuit, and read the display.",
        "icon_url": "https://img.icons8.com/color/96/multimeter.png",
        "safety_tips": "Never measure resistance on a live circuit; always start with the highest range and work down."
    },
    "Pullers": {
        "name": "Bearing / Gear Puller",
        "description": "Removes bearings, gears, and pulleys from shafts.",
        "usage": "Attach the puller arms around the component, centre the screw on the shaft, and tighten gradually.",
        "icon_url": "https://img.icons8.com/color/96/puller.png",
        "safety_tips": "Ensure the puller is centred to avoid bending the shaft; use a protective barrier in case of sudden release."
    },
    "Vernier Calipers": {
        "name": "Vernier Caliper",
        "description": "Precision measuring tool for internal/external dimensions and depth.",
        "usage": "Slide the jaws to the object, read the main scale plus the vernier scale for fine measurements.",
        "icon_url": "https://img.icons8.com/color/96/caliper.png",
        "safety_tips": "Do not force the jaws; clean the measuring surfaces before use; store in a protective case."
    }
}
