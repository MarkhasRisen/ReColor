from backend.app.pipeline.profile import VisionProfile


def test_profile_from_ishihara_infers_protan():
    responses = {"p1": "incorrect", "p2": "incorrect", "p3": "correct"}
    profile = VisionProfile.from_ishihara_results(responses)
    assert profile.deficiency in {"protan", "normal"}
