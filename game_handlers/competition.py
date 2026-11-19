"""Level competition: two players take turns on one level, each a single attempt, and the higher score wins it."""
from typing import List, Optional


class Competition:
    """One level's pair of turns, and the points so far. `first` is the player (0 or 1) who flies first.

    A player's score is what the ship has left when it goes through the portal; a ship that runs out of power scores
    zero. Both at zero means nobody scored, and a campaign replays the level."""

    def __init__(self, level: int, from_levels: bool, first: int = 0, points: Optional[List[int]] = None,
                 rounds_played: int = 0):
        self.level = level
        self.from_levels = from_levels      # picked from LEVELS: the pair play it once and return to the list
        self.first = first
        self.points = list(points) if points is not None else [0, 0]
        self.rounds_played = rounds_played
        self.scores: List[Optional[float]] = [None, None]
        self.breakdowns: List[Optional[dict]] = [None, None]   # each player's score components, for the results
        self.turn = 0                       # 0 and 1: the first and second flights; 2 once both have flown

    @property
    def order(self):
        return (self.first, 1 - self.first)

    @property
    def current(self) -> Optional[int]:
        """The player whose turn it is, or None once both have flown."""
        return self.order[self.turn] if self.turn < 2 else None

    @property
    def finished(self) -> bool:
        return self.turn >= 2

    def record(self, score: float, breakdown: Optional[dict] = None) -> None:
        """The player now flying finishes their turn with `score`; the other follows, or the level is done."""
        self.scores[self.current] = max(0.0, score)
        self.breakdowns[self.current] = breakdown
        self.turn += 1
        if self.finished:
            winner = self.winner
            if winner is not None:
                self.points[winner] += 1

    @property
    def winner(self) -> Optional[int]:
        """The player with the higher score, or None for a tie (which includes both scoring nothing)."""
        if not self.finished or self.scores[0] == self.scores[1]:
            return None
        return 0 if self.scores[0] > self.scores[1] else 1

    @property
    def both_out(self) -> bool:
        return self.finished and all(score == 0 for score in self.scores)

    def next_round(self) -> Optional['Competition']:
        """What follows a finished level: nothing from LEVELS; in a campaign the same level again when both ran out of
        power, else the next level with the other player flying first."""
        if self.from_levels:
            return None
        if self.both_out:
            return Competition(self.level, False, self.first, self.points, self.rounds_played)
        return Competition(self.level + 1, False, 1 - self.first, self.points, self.rounds_played + 1)
