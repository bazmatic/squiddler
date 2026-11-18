"""Campaign-only practice levels, built from the real level rather than added to the campaign."""
from dataclasses import replace

EXIT_WITH_CRAWLER = 1032


def exit_with_crawler_layout(layout):
    """Level 32's final passage: the ship near the open exit, a lone crawler behind walls."""
    if layout is None:
        raise ValueError('Campaign practice level 1032 requires the authored level 32 layout')
    return replace(layout, start=(20, 28), enemies=(('wallcrawler', (28, 17)),))
