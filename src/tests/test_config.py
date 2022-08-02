import IAFautoclass.Config as Config

def test_defaults():
    config = Config.Config()

    assert config.name == 'iris'
