# ripeness_database.py

FRESHNESS_DATA = {
    "strawberry": {
        "unripe": {
            "days_remaining": 6,
            "description": "White or pale red, hard texture",
            "storage_tips": "Leave at room temperature for 1-2 days to ripen",
            "recipes": ["Wait to ripen - will be sour if eaten now"],
            "action": "Wait"
        },
        "fresh": {
            "days_remaining": 3,
            "description": "Bright red, firm, green leaves intact",
            "storage_tips": "Refrigerate unwashed in original container",
            "recipes": ["Eat fresh", "Smoothies", "Fruit salad", "Yogurt topping"],
            "action": "Eat soon"
        },
        "bruised_overripe": {
            "days_remaining": 1,
            "description": "Dark spots, soft texture, but no mold",
            "storage_tips": "Use today or freeze (no more than 2 days) to avoid mold",
            "recipes": ["Jam", "Smoothies", "Baked goods", "Compote"],
            "action": "Cook/Process NOW"
        },
        "spoiled": {
            "days_remaining": 0,
            "description": "Mold visible, slimy texture",
            "storage_tips": "Discard immediately",
            "recipes": ["Not safe to eat"],
            "action": "Discard"
        }
    },
    
    "spinach": {
        "fresh": {
            "days_remaining": 7,
            "description": "Crisp leaves, vibrant green, no wilting",
            "storage_tips": "Refrigerate in crisper drawer, wrapped in paper towel (avoid humidity)",
            "recipes": ["Salads", "Smoothies", "Sautéed", "Raw"],
            "action": "Use within week"
        },
        "aging": {
            "days_remaining": 2,
            "description": "Slight wilting, edges browning",
            "storage_tips": "Use immediately, cook rather than eat raw",
            "recipes": ["Cooked in soup", "Sautéed", "Smoothies", "Pasta"],
            "action": "Cook today (try not to eat raw)"
        },
        "expired": {
            "days_remaining": 0,
            "description": "Completely slimy, yellow/brown, bad odor",
            "storage_tips": "Discard - not safe to eat",
            "recipes": ["Not safe"],
            "action": "Discard"
        }
    },
    
    "tomato": {
        "unripe": {
            "days_remaining": 7,
            "description": "Green or pale red, very firm",
            "storage_tips": "Leave at room temp in paper bag to ripen faster",
            "recipes": ["Wait to ripen", "Green tomato recipes if desired"],
            "action": "Wait to ripen"
        },
        "fresh": {
            "days_remaining": 5,
            "description": "Uniform red, firm but yields very slightly to pressure",
            "storage_tips": "Store at room temp until cut, then refrigerate",
            "recipes": ["Salads", "Sandwiches", "Eat fresh", "Salsa"],
            "action": "Eat within 5 days"
        },
        "overripe_eatable": {
            "days_remaining": 1,
            "description": "Very soft, wrinkled skin, but no mold",
            "storage_tips": "Use today for cooking or freeze for sauces",
            "recipes": ["Tomato sauce", "Soup", "Roasted", "Cooked dishes"],
            "action": "Cook TODAY, could be too mushy and not pleasant to eat raw"
        },
        "spoiled": {
            "days_remaining": 0,
            "description": "Mold, leaking, foul smell",
            "storage_tips": "Discard",
            "recipes": ["Not safe"],
            "action": "Discard"
        }
    },
    
    "banana": {
        "underripe": {
            "days_remaining": 7,
            "description": "Green tips, very firm",
            "storage_tips": "Leave at room temp to ripen (3-5 days)",
            "recipes": ["Wait to ripen - will be starchy"],
            "action": "Wait"
        },
        "fresh": {
            "days_remaining": 5,
            "description": "Bright yellow, no spots, firm",
            "storage_tips": "Cooler temperatures, separate from other produce",
            "recipes": ["Eat fresh", "Smoothies", "Cereal topping"],
            "action": "Enjoy fresh and keep away from stooves and apples (they speed up ripening)!"
        },
        "medium": {
            "days_remaining": 3,
            "description": "Yellow with small brown spots, slightly soft",
            "storage_tips": "Eat soon or refrigerate to slow ripening",
            "recipes": ["Smoothies", "Oatmeal", "Pancakes", "Still great fresh!"],
            "action": "Eat within 2-3 days keep far from apples and stooves (very warm places)"
        },
        "overripe_bread": {
            "days_remaining": 1,
            "description": "Many brown spots or mostly brown, very soft, BUT PERFECT FOR BAKING! 🍌🍞",
            "storage_tips": "Freeze if not using today (great for smoothies later!)",
            "recipes": ["BANANA BREAD (the best!)", "Muffins", "Pancakes", "Nice cream", "Freeze for smoothies"],
            "action": "Bake TODAY or freeze!"
        },
        "spoiled": {
            "days_remaining": 0,
            "description": "Black, leaking liquid, mold, foul smell, wrinkled skin",
            "storage_tips": "Discard",
            "recipes": ["Too far gone"],
            "action": "Discard"
        }
    },
    
    "avocado": {
        "unripe": {
            "days_remaining": 5,
            "description": "Hard, bright green, doesn't yield to pressure",
            "storage_tips": "Room temp to ripen, put in paper bag with apple to speed up",
            "recipes": ["Wait 3-5 days to ripen"],
            "action": "Wait"
        },
        "fresh": {
            "days_remaining": 3,
            "description": "Yields slightly to pressure, darker green",
            "storage_tips": "Almost ready! Check daily",
            "recipes": ["Almost ready for guacamole"],
            "action": "Check tomorrow"
        },
        "overripe_eatable": {
            "days_remaining": 1,
            "description": "Yields easily to gentle pressure, dark green/brown",
            "storage_tips": "Eat today! Refrigerate if waiting 1 day",
            "recipes": ["Guacamole", "Toast", "Salads", "Smoothies"],
            "action": "EAT TODAY!"
        },
        "spoiled": {
            "days_remaining": 0,
            "description": "Very soft, dark spots inside",
            "storage_tips": "Use immediately if no brown spots inside",
            "recipes": ["Smoothies", "Baking", "Chocolate mousse"],
            "action": "Use NOW or discard"
        }
    }
}

# Fun facts for engagement
FOOD_WASTE_FACTS = {
    "strawberry": "Strawberries are one of the most wasted fruits - 25% never get eaten!",
    "spinach": "Leafy greens account for 12% of household food waste",
    "tomato": "Average household wastes 3 tomatoes per month",
    "banana": "Bananas are the #1 wasted fruit globally, with more than 3.5 million tons/year!",
    "avocado": "Almost 50% of avocados are thrown away because people misjudge ripeness"
}