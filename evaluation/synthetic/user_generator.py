"""Generate 100 deterministic synthetic users — Deliverable 1."""

from __future__ import annotations

import random
from typing import Any

from evaluation.synthetic.conversation_generator import generate_conversation
from evaluation.synthetic.event_generator import (
    build_emotion_trajectory,
    generate_temporal_events,
)
from evaluation.synthetic.synthetic_user import SyntheticUser
from evaluation.synthetic.user_profile import (
    EmotionalProfile,
    PersonalityTraits,
    Preference,
    Project,
    Relationship,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Corpus pools — all deterministic
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Quinn", "Avery",
    "Sage", "Rowan", "Finley", "Blake", "Skyler", "Dakota", "Reese", "Parker",
    "Logan", "Drew", "Cameron", "Jamie", "Kendall", "Bailey", "Peyton", "Hayden",
    "Emery", "River", "Phoenix", "Remi", "Oakley", "Elliot", "Sasha", "Zuri",
    "Kai", "Ari", "Noel", "Eden", "Sol", "Noa", "Luca", "Nico",
]

_LAST_NAMES = [
    "Chen", "Patel", "Williams", "Okonkwo", "Martinez", "Rosenberg", "Kim",
    "Santos", "Nakamura", "Fischer", "Andersen", "Petrov", "Gupta", "Diallo",
    "Moreau", "Johansson", "Mensah", "Kobayashi", "Ferreira", "Nielsen",
    "Svensson", "Lindqvist", "Osei", "Ito", "Yamamoto", "Leblanc", "Dupont",
    "Müller", "Schmidt", "Bauer", "Kowalski", "Novak", "Popescu", "Vasquez",
]

_OCCUPATIONS = [
    "Software Engineer", "Machine Learning Researcher", "Data Scientist",
    "PhD Student", "Research Engineer", "Backend Developer", "Full-Stack Engineer",
    "Computer Vision Engineer", "NLP Researcher", "Systems Engineer",
    "Product Engineer", "Site Reliability Engineer", "Security Researcher",
    "Biomedical Informatics Researcher", "Cognitive Science PhD Student",
]

_WORKPLACES = [
    "DeepMind", "Anthropic", "OpenAI", "Google Brain", "Meta AI", "Microsoft Research",
    "CMU School of Computer Science", "MIT CSAIL", "Stanford AI Lab",
    "Berkeley BAIR", "ETH Zurich AI Center", "Oxford Future of Humanity Institute",
    "Nvidia Research", "Apple ML", "Amazon Science", "IBM Research",
    "University of Toronto", "University of Edinburgh", "TU Berlin",
]

_LOCATIONS = [
    "San Francisco, CA", "Cambridge, MA", "New York, NY", "London, UK",
    "Toronto, Canada", "Berlin, Germany", "Paris, France", "Tokyo, Japan",
    "Singapore", "Zurich, Switzerland", "Montreal, Canada", "Amsterdam, Netherlands",
    "Seattle, WA", "Austin, TX", "Boston, MA",
]

_TOPICS = [
    "reinforcement learning", "transformer architectures", "graph neural networks",
    "probabilistic inference", "causal reasoning", "computer vision",
    "natural language processing", "multi-agent systems", "robotics",
    "federated learning", "continual learning", "meta-learning",
    "interpretability", "alignment", "cognitive architectures",
    "knowledge graphs", "semantic search", "distributed systems",
    "software architecture", "DevOps", "type theory", "formal verification",
]

_EDITORS = [
    "VSCode", "Neovim", "IntelliJ IDEA", "Emacs", "PyCharm", "Zed", "Helix",
]

_LANGUAGES = [
    "Python", "Rust", "TypeScript", "Haskell", "Julia", "Go", "Scala", "Kotlin",
]

_ARCHITECTURES = [
    "microservices", "event-driven", "monolith", "CQRS", "serverless", "hexagonal",
]

_HOBBIES = [
    "rock climbing", "chess", "running", "cooking", "reading science fiction",
    "playing guitar", "hiking", "photography", "board games", "swimming",
    "sketching", "cycling", "yoga", "meditation", "pottery",
    "learning new languages", "bouldering", "baking", "gardening", "gaming",
]

_GOALS = [
    "lead a research team at a top AI lab",
    "publish a landmark paper in cognitive science",
    "build an open-source project with 10k stars",
    "complete a PhD in machine learning",
    "transition into entrepreneurship",
    "become a principal engineer",
    "write a technical book on distributed systems",
    "start an AI-focused nonprofit",
    "improve human-computer interaction through better cognitive models",
    "contribute meaningfully to AI safety research",
]

_PERSON_NAMES = [
    "Alice", "Bob", "Carlos", "Diana", "Ethan", "Fiona", "George",
    "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Marcus", "Nina",
    "Oscar", "Paula", "Rafael", "Sara", "Thomas", "Uma",
]

_RELATIONSHIP_TYPES = [
    "mentor", "collaborator", "peer", "manager", "direct-report",
    "co-author", "friend", "advisor",
]

_EDUCATION_BY_ROLE = {
    "research": [
        "MS in Machine Learning from University of Toronto",
        "PhD coursework in Cognitive Science at CMU",
        "MSc in Statistics from ETH Zurich",
    ],
    "engineering": [
        "BS in Computer Science from Georgia Tech",
        "MS in Software Engineering from TU Berlin",
        "self-directed systems engineering background",
    ],
    "academic": [
        "PhD track in Machine Learning",
        "doctoral training in Human-Computer Interaction",
        "MS in Computational Neuroscience",
    ],
}

_COMMUNICATION_STYLES = [
    "concise and evidence-first",
    "reflective with careful caveats",
    "warm, narrative, and context-heavy",
    "direct with implementation details up front",
    "question-driven and exploratory",
]

_HABITS = [
    "writes a daily research log",
    "blocks mornings for deep work",
    "reviews notes every Friday",
    "keeps a running decision journal",
    "does weekly project retrospectives",
    "uses calendar timeboxing",
]

_FINANCIAL_GOALS = [
    "build a six-month emergency fund",
    "save for a sabbatical year",
    "pay down student loans",
    "invest consistently in index funds",
    "set aside money for family support",
]

_TRAVEL_PLACES = [
    "Vancouver", "Lisbon", "Bangalore", "Copenhagen", "Seoul", "Dublin",
    "Barcelona", "Melbourne", "Helsinki", "Taipei",
]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _build_preferences(rng: random.Random, total_turns: int) -> list[Preference]:
    """Generate evolving preference chains."""
    prefs: list[Preference] = []

    # Editor evolution: 2–3 editor switches
    editors = rng.sample(_EDITORS, min(3, len(_EDITORS)))
    # Guard: need at least one valid turn in (20, total_turns-20) for switches
    switchable_range_size = max(0, total_turns - 41)  # range(20, total_turns-20) length
    n_switches = min(2, switchable_range_size, total_turns // 5)
    if n_switches > 0:
        switch_turns = sorted(rng.sample(range(20, total_turns - 20), n_switches))
    else:
        switch_turns = []
    switch_turns = [0] + switch_turns
    for idx, turn in enumerate(switch_turns):
        superseded_at = switch_turns[idx + 1] if idx + 1 < len(switch_turns) else -1
        next_editor = editors[idx % len(editors)]
        prefs.append(
            Preference(
                category="editor",
                value=next_editor,
                turn_set=turn,
                superseded_by=editors[(idx + 1) % len(editors)] if superseded_at >= 0 else "",
                superseded_at_turn=superseded_at,
            )
        )

    # Language preference — stable
    lang = rng.choice(_LANGUAGES)
    prefs.append(Preference(category="language", value=lang, turn_set=0))

    # Architecture preference — one change
    arch1 = rng.choice(_ARCHITECTURES)
    arch_switch = rng.randint(total_turns // 3, 2 * total_turns // 3) if total_turns > 60 else -1
    prefs.append(Preference(category="architecture", value=arch1, turn_set=0, superseded_at_turn=arch_switch))
    if arch_switch > 0:
        arch2 = rng.choice([a for a in _ARCHITECTURES if a != arch1])
        prefs.append(Preference(category="architecture", value=arch2, turn_set=arch_switch))

    return prefs


def _build_relationships(rng: random.Random, total_turns: int) -> list[Relationship]:
    """Generate relationships that evolve over the conversation arc."""
    relationships: list[Relationship] = []
    n_people = rng.randint(3, 6)
    people = rng.sample(_PERSON_NAMES, n_people)

    spacing = max(1, total_turns // (n_people + 1))
    for idx, person in enumerate(people):
        rel_type = rng.choice(_RELATIONSHIP_TYPES)
        met_at = spacing * idx
        # Some relationships end
        duration = rng.randint(spacing, min(spacing * 3, total_turns))
        ended_at = met_at + duration if rng.random() < 0.4 else -1
        if ended_at >= total_turns:
            ended_at = -1
        relationships.append(
            Relationship(
                person_name=person,
                relationship_type=rel_type,
                status="active" if ended_at < 0 else "ended",
                met_at_turn=met_at,
                ended_at_turn=ended_at,
                context=rng.choice(["research collaboration", "team project", "conference", "internship", "class"]),
                evolution_history=[],
            )
        )
    return relationships


def _build_projects(rng: random.Random, topics: list[str]) -> list[Project]:
    """Generate 2–4 projects."""
    n = rng.randint(2, 4)
    project_names = [f"Project-{rng.choice(topics).replace(' ', '_').title()}" for _ in range(n)]
    projects: list[Project] = []
    for name in project_names:
        topic = rng.choice(topics)
        projects.append(
            Project(
                name=name,
                description=f"A project focused on {topic}.",
                status=rng.choice(["active", "active", "completed"]),
                started_at_turn=rng.randint(0, 20),
                milestones=[f"Milestone {i}" for i in range(1, rng.randint(2, 5))],
                collaborators=rng.sample(_PERSON_NAMES, rng.randint(1, 3)),
            )
        )
    return projects


def _role_family(occupation: str) -> str:
    occ = occupation.lower()
    if any(term in occ for term in ("phd", "student", "science")):
        return "academic"
    if any(term in occ for term in ("research", "scientist", "informatic")):
        return "research"
    return "engineering"


def _build_rich_profile_context(
    rng: random.Random,
    *,
    occupation: str,
    topics: list[str],
    hobbies: list[str],
    relationships: list[Relationship],
    projects: list[Project],
    total_turns: int,
) -> dict[str, Any]:
    """Create correlated life-history fields instead of independent random facts."""
    role = _role_family(occupation)
    research_area = topics[0] if topics else "human-centered AI"
    colleagues = [r.person_name for r in relationships if r.relationship_type not in {"friend"}][:4]
    friends = [r.person_name for r in relationships if r.relationship_type == "friend"][:3]
    if not friends:
        friends = rng.sample(_PERSON_NAMES, 2)

    family_label = rng.choice(["sister", "brother", "parent", "partner", "cousin"])
    family = [f"{rng.choice(_PERSON_NAMES)} ({family_label})"]
    tech_stack = [rng.choice(_LANGUAGES), rng.choice(_EDITORS), rng.choice(_ARCHITECTURES)]
    travel_history = rng.sample(_TRAVEL_PLACES, rng.randint(1, 3))
    health_context = rng.sample(
        [
            "manages stress with running",
            "protects sleep before deadlines",
            "uses meditation after intense review cycles",
            "takes short walks between coding blocks",
        ],
        2,
    )
    career_changes = [
        {
            "turn": max(1, total_turns // 4),
            "from": "individual contributor",
            "to": occupation,
            "reason": f"wanted deeper work in {research_area}",
        }
    ]
    milestones = [
        {"turn": p.started_at_turn, "label": f"started {p.name}", "type": "project"}
        for p in projects[:3]
    ]
    relationship_timeline = [
        {
            "turn": rel.met_at_turn,
            "person": rel.person_name,
            "relationship_type": rel.relationship_type,
            "status": rel.status,
            "context": rel.context,
        }
        for rel in relationships
    ]

    return {
        "research_area": research_area,
        "education": rng.choice(_EDUCATION_BY_ROLE[role]),
        "interests": list(dict.fromkeys([*topics[:3], *hobbies[:2]])),
        "family": family,
        "friends": friends,
        "colleagues": colleagues,
        "communication_style": rng.choice(_COMMUNICATION_STYLES),
        "tech_stack": tech_stack,
        "travel_history": travel_history,
        "health_context": health_context,
        "financial_goals": rng.sample(_FINANCIAL_GOALS, 2),
        "relationship_timeline": relationship_timeline,
        "career_changes": career_changes,
        "milestones": milestones,
        "habits": rng.sample(_HABITS, 3),
    }


class UserGenerator:
    """Generate N deterministic synthetic users, each with a unique seed."""

    DEFAULT_SEED_BASE = 42

    def __init__(
        self,
        n_users: int = 100,
        min_turns: int = 500,
        max_turns: int = 1000,
        seed_base: int = DEFAULT_SEED_BASE,
    ) -> None:
        self.n_users = n_users
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.seed_base = seed_base

    def generate_all(self) -> list[SyntheticUser]:
        """Generate all synthetic users deterministically."""
        return [self.generate_user(i) for i in range(self.n_users)]

    def generate_user(self, user_index: int) -> SyntheticUser:
        """Generate a single synthetic user by index (deterministic)."""
        seed = self.seed_base + user_index * 1000
        rng = random.Random(seed)  # nosec B311 — deterministic simulation seed, not security-critical

        user_id = f"user_{user_index:04d}"
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        name = f"{first} {last}"
        occupation = rng.choice(_OCCUPATIONS)
        workplace = rng.choice(_WORKPLACES)
        location = rng.choice(_LOCATIONS)
        total_turns = rng.randint(self.min_turns, self.max_turns)
        topics = rng.sample(_TOPICS, rng.randint(3, 6))
        hobbies = rng.sample(_HOBBIES, rng.randint(2, 5))
        goals = rng.sample(_GOALS, rng.randint(2, 4))

        preferences = _build_preferences(rng, total_turns)
        relationships = _build_relationships(rng, total_turns)
        projects = _build_projects(rng, topics)
        rich_context = _build_rich_profile_context(
            rng,
            occupation=occupation,
            topics=topics,
            hobbies=hobbies,
            relationships=relationships,
            projects=projects,
            total_turns=total_turns,
        )
        known_facts: dict[str, str] = {
            "name": name,
            "occupation": occupation,
            "workplace": workplace,
            "location": location,
            "research_area": str(rich_context["research_area"]),
            "education": str(rich_context["education"]),
            "communication_style": str(rich_context["communication_style"]),
        }

        # Personality — Big Five
        traits = PersonalityTraits(
            openness=round(rng.uniform(0.3, 0.9), 2),
            conscientiousness=round(rng.uniform(0.4, 0.95), 2),
            extraversion=round(rng.uniform(0.2, 0.8), 2),
            agreeableness=round(rng.uniform(0.3, 0.9), 2),
            neuroticism=round(rng.uniform(0.1, 0.7), 2),
        )

        # Temporal events
        profile_hints = {
            "name": name, "workplace": workplace, "occupation": occupation,
            "location": location, "topic": topics[0],
            "project": projects[0].name if projects else "main_project",
            "hobby": hobbies[0] if hobbies else "reading",
        }
        temporal_events = generate_temporal_events(rng, total_turns, occupation, profile_hints)

        # Emotion trajectory
        baseline_emotions = [
            "joy", "anticipation", "trust", "neutral", "anticipation",
        ]
        baseline = rng.choice(baseline_emotions)
        traj = build_emotion_trajectory(rng, total_turns, temporal_events, baseline)
        emotional_profile = EmotionalProfile(
            baseline_emotion=baseline,
            current_emotion=traj[-1][1] if traj else baseline,
            trajectory=traj,
            volatility=round(traits.neuroticism * 0.6 + 0.1, 2),
            recovery_rate=round((1 - traits.neuroticism) * 0.7 + 0.2, 2),
        )

        profile = UserProfile(
            user_id=user_id,
            name=name,
            occupation=occupation,
            workplace=workplace,
            location=location,
            hobbies=hobbies,
            long_term_goals=goals,
            recurring_topics=topics,
            preferences=preferences,
            projects=projects,
            relationships=relationships,
            temporal_events=temporal_events,
            emotional_profile=emotional_profile,
            personality_traits=traits,
            known_facts=known_facts,
            research_area=str(rich_context["research_area"]),
            education=str(rich_context["education"]),
            interests=list(rich_context["interests"]),
            family=list(rich_context["family"]),
            friends=list(rich_context["friends"]),
            colleagues=list(rich_context["colleagues"]),
            communication_style=str(rich_context["communication_style"]),
            tech_stack=list(rich_context["tech_stack"]),
            travel_history=list(rich_context["travel_history"]),
            health_context=list(rich_context["health_context"]),
            financial_goals=list(rich_context["financial_goals"]),
            relationship_timeline=list(rich_context["relationship_timeline"]),
            career_changes=list(rich_context["career_changes"]),
            milestones=list(rich_context["milestones"]),
            habits=list(rich_context["habits"]),
        )

        # Generate conversation
        turns, id_probes, pref_probes, fact_probes, emo_probes = generate_conversation(
            profile, rng, total_turns, temporal_events
        )

        return SyntheticUser(
            user_id=user_id,
            name=name,
            seed=seed,
            profile=profile,
            conversation_turns=turns,
            identity_probes=id_probes,
            preference_probes=pref_probes,
            fact_probes=fact_probes,
            emotion_probes=emo_probes,
        )

    def to_summary(self, users: list[SyntheticUser]) -> dict[str, Any]:
        """Produce a JSON-serialisable summary of the user cohort."""
        return {
            "total_users": len(users),
            "seed_base": self.seed_base,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "users": [
                {
                    "user_id": u.user_id,
                    "name": u.name,
                    "seed": u.seed,
                    "total_turns": u.total_turns,
                    "n_preferences": len(u.preferences),
                    "n_projects": len(u.projects),
                    "n_relationships": len(u.relationships),
                    "n_events": len(u.temporal_events),
                    "baseline_emotion": u.emotional_profile.baseline_emotion,
                }
                for u in users
            ],
        }
