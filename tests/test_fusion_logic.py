from soundguard.emergency.fusion_engine import fuse_result


def run_tests():
    cases = [
        (
            "case1",
            "giúp tôi với",
            {"label": "Siren", "confidence": 0.80},
            {"matched": True, "category": "siren", "message_vi": "Còi báo động", "is_dangerous": True, "confidence": 0.80},
            "CRITICAL",
        ),
        (
            "case2",
            "",
            {"label": "Glass breaking", "confidence": 0.70},
            {"matched": True, "category": "glass breaking", "message_vi": "Kính vỡ", "is_dangerous": True, "confidence": 0.70},
            "HIGH",
        ),
        (
            "case3",
            "",
            {"label": "Bark", "confidence": 0.65},
            {"matched": True, "category": "bark", "message_vi": "Tiếng chó sủa", "is_dangerous": True, "confidence": 0.65},
            "LOW",
        ),
        (
            "case4",
            "Một ngày bình thường",
            {"label": "Speech", "confidence": 0.90},
            {"matched": True, "category": "speech", "message_vi": "Speech", "is_dangerous": False, "confidence": 0.90},
            "LOW",
        ),
        (
            "case5",
            "",
            {"label": "Siren", "confidence": 0.20},
            {"matched": False, "category": "low_confidence", "message_vi": "", "is_dangerous": False, "confidence": 0.20},
            "LOW",
        ),
    ]

    for name, transcript, sound_result, alert_result, expected_level in cases:
        result = fuse_result("test", transcript, sound_result, alert_result)
        print(f"{name}: {result['alert_level']} => expected {expected_level}")
        assert result["alert_level"] == expected_level, (name, result)

    print("fusion tests passed")


if __name__ == "__main__":
    run_tests()
