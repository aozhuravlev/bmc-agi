"""
BMC Node Definitions — 505 memes in 21 clusters
=================================================
Scaled-up meme graph for memplex_visualization simulation.
Original 47 memes (5 clusters) → 505 memes (21 clusters).

Exports:
    MEME_CLUSTERS       — OrderedDict: cluster_name → [meme_names]
    HUB_MEMES           — set of hub meme names
    NEGATIVE_EDGES       — list of (m1, m2, weight) tuples
    CROSS_LINKS          — list of (m1, m2, weight) tuples
    UTILITY_CONNECTIONS  — dict: utility_name → [(meme, weight, etype)]
    INCOMPAT_SPEC        — dict: utility_name → {meme: value}
"""

from collections import OrderedDict

# ============================================================
# MEME CLUSTERS — 505 bio-relevant memes in 21 clusters
# ============================================================

MEME_CLUSTERS = OrderedDict([

    # ── Expanded original clusters (5 × ~28 = ~140) ──

    ('profession', [
        'Programming', 'Algorithms', 'Career_growth', 'Deadline_pressure',
        'Code_quality', 'Open_source', 'Technical_writing', 'Mentoring',
        # new (17)
        'Project_management', 'Code_review', 'System_design', 'Debugging_skill',
        'Version_control', 'Documentation', 'Pair_programming', 'Agile_method',
        'Testing_practice', 'DevOps_culture', 'Tech_leadership', 'Remote_work',
        'Startup_culture', 'Corporate_ladder',
        'Freelancing', 'Domain_expertise', 'Architecture_skill',
    ]),

    ('family', [
        'Parenting', 'Marriage_vows', 'Family_dinner', 'Child_education',
        'Elderly_care', 'Home_comfort', 'Traditions',
        # new (18)
        'Sibling_bond', 'Extended_family', 'Family_vacation', 'Bedtime_ritual',
        'Homework_help', 'Family_budget', 'Chore_sharing', 'Family_photo',
        'Holiday_celebration', 'Family_recipe', 'Parent_sacrifice', 'Family_pride',
        'Grandparent_wisdom', 'Adoption_openness', 'Blended_family',
        'Inheritance_norm', 'Family_business', 'Childhood_memory',
    ]),

    ('beliefs', [
        'Free_will', 'Rationalism', 'Humanism', 'Science_trust',
        'Meritocracy', 'Self_improvement', 'Democracy',
        'Environmental_ethics', 'Stoicism',
        # new (16)
        'Moral_framework', 'Worldview', 'Justice_belief', 'Progress_belief',
        'Karma_belief', 'Truth_commitment', 'Pragmatism', 'Relativism',
        'Core_identity', 'Human_nature_view', 'Tradition_value', 'Innovation_value',
        'Conspiracy_openness', 'Fate_belief', 'Social_contract', 'Utilitarianism',
    ]),

    ('hobbies', [
        'Chess', 'Running', 'Reading_fiction', 'Cooking',
        'Photography', 'Music_listening', 'Board_games',
        # new (18)
        'Gardening', 'Painting', 'Yoga_practice', 'Hiking',
        'Bird_watching', 'Knitting', 'Woodworking', 'Fishing',
        'Cycling', 'Video_gaming', 'Podcasting', 'Journaling',
        'Puzzle_solving', 'Calligraphy', 'Wine_tasting', 'Stargazing',
        'Rock_climbing', 'Martial_arts',
    ]),

    ('knowledge', [
        'Neuroscience_basics', 'Evolution_theory', 'Graph_theory',
        'Meme_theory', 'Statistical_thinking', 'History_WW2',
        'Philosophy_mind', 'Complexity_science',
        # new (17)
        'Quantum_basics', 'Climate_science', 'Genetics_knowledge', 'AI_understanding',
        'Economics_101', 'Psychology_basics', 'Sociology_basics', 'Linguistics_knowledge',
        'Astronomy_basics', 'Probability_theory', 'Information_theory',
        'Game_theory', 'Network_science', 'Cognitive_science',
        'Ethics_theory', 'Systems_thinking', 'Critical_thinking',
    ]),

    # ── New clusters (15 clusters, ~360 memes) ──

    ('language', [
        'Native_language', 'Grammar_rules', 'Vocabulary_depth',
        'Narrative_skill', 'Pragmatics', 'Metalinguistic_awareness',
        'Reading_habit', 'Writing_skill', 'Bilingual_identity',
        'Dialect_awareness', 'Slang_usage', 'Humor_style',
        'Debate_skill', 'Storytelling', 'Public_speaking',
        'Listening_skill', 'Poetry_appreciation', 'Jargon_mastery',
        'Translation_skill', 'Code_switching',
    ]),

    ('social_norms', [
        'Social_hierarchy', 'Reciprocity', 'Fairness_norm', 'Punishment_norm',
        'Cooperation_norm', 'Reputation_management', 'Gender_roles', 'Taboos',
        'Privacy_norm', 'Hospitality_norm', 'Gift_giving',
        'Queue_etiquette', 'Personal_space', 'Age_respect',
        'Authority_deference', 'Dress_code', 'Table_manners',
        'Conflict_resolution', 'Apology_norm', 'Tipping_norm',
        'Sharing_norm', 'Honor_code', 'Whistleblower_norm',
        'Bystander_norm',
    ]),

    ('skills', [
        'Tool_use', 'Cooking_skill', 'Navigation', 'Planning_ahead',
        'Risk_assessment', 'Impulse_delay', 'Resource_management',
        'Driving_skill', 'Swimming_skill', 'First_aid',
        'DIY_repair', 'Gardening_skill', 'Negotiation',
        'Time_management', 'Multitasking', 'Critical_analysis',
        'Problem_solving', 'Digital_literacy', 'Financial_skill',
        'Teaching_skill', 'Leadership_skill', 'Decision_making',
        'Budgeting_skill', 'Networking_skill',
    ]),

    ('emotion_regulation', [
        'Anger_management', 'Fear_coping', 'Grief_processing', 'Joy_expression',
        'Attachment_style', 'Empathy_skill', 'Stress_response',
        'Mindfulness_practice', 'Humor_coping', 'Emotional_expression',
        'Suppression_habit', 'Emotional_granularity', 'Self_soothing',
        'Boundary_setting', 'Resilience', 'Optimism_bias',
        'Catastrophizing', 'Gratitude_practice', 'Forgiveness_skill',
        'Jealousy_management', 'Shame_coping', 'Guilt_processing',
        'Loneliness_management', 'Patience_skill',
    ]),

    ('identity', [
        'Group_identity', 'Cultural_pride', 'Personal_narrative', 'Role_identity',
        'Status_symbol', 'Ethnic_identity', 'Generational_values',
        'Professional_identity', 'Parent_identity', 'Age_identity',
        'Body_identity', 'Intellectual_identity', 'Creative_identity',
        'National_identity', 'Regional_identity', 'Class_identity',
        'Activist_identity', 'Hobbyist_identity', 'Digital_identity',
        'Sports_identity', 'Spiritual_identity', 'Gender_identity',
        'Immigrant_identity', 'Veteran_identity',
    ]),

    ('technology', [
        'Smartphone_dependence', 'Social_media_habit', 'AI_attitude',
        'Privacy_concern', 'Digital_detox', 'Online_shopping',
        'Streaming_habit', 'Gaming_culture', 'Coding_interest',
        'Automation_fear', 'Tech_optimism', 'Internet_slang',
        'Meme_culture', 'Cybersecurity_awareness', 'VR_openness',
        'Crypto_attitude', 'Open_source_value', 'Smart_home',
        'Wearable_tech', 'Cloud_reliance', 'Algorithm_awareness',
        'Data_privacy', 'Tech_minimalism',
        'Robot_attitude',
    ]),

    ('politics', [
        'Democracy_value', 'Authority_respect', 'Political_identity',
        'Voting_habit', 'Protest_attitude', 'Free_speech_defense',
        'Equality_drive', 'Nationalism', 'Globalism',
        'Tax_attitude', 'Welfare_attitude', 'Military_attitude',
        'Immigration_stance', 'Law_respect', 'Political_cynicism',
        'Activism_drive', 'Censorship_attitude', 'Corruption_tolerance',
        'Diplomacy_value', 'Lobby_awareness',
        'Transparency_demand', 'Civil_disobedience', 'Populism_attitude',
        'Technocracy_view',
    ]),

    ('economics', [
        'Work_ethic', 'Saving_habit', 'Consumption_pattern',
        'Debt_attitude', 'Career_ambition', 'Entrepreneurship',
        'Wealth_attitude', 'Property_value', 'Investment_interest',
        'Frugality', 'Luxury_attitude', 'Brand_loyalty',
        'Bargain_hunting', 'Side_hustle', 'Retirement_planning',
        'Insurance_trust', 'Charity_giving', 'Union_attitude',
        'Gig_economy_view', 'Inflation_anxiety',
        'UBI_attitude', 'Fair_trade_value', 'Stock_market_trust',
        'Cooperative_model',
    ]),

    ('health', [
        'Exercise_habit', 'Nutrition_belief', 'Body_image',
        'Sleep_hygiene', 'Vaccination_trust', 'Mental_health_awareness',
        'Addiction_awareness', 'Doctor_trust', 'Alternative_medicine',
        'Hygiene_standard', 'Pain_tolerance', 'Disability_awareness',
        'Organ_donation_view', 'Aging_attitude', 'Fitness_identity',
        'Diet_ideology', 'Supplement_belief', 'Therapy_openness',
        'Preventive_care', 'Hydration_habit', 'Posture_awareness',
        'Gut_health_belief', 'Genetic_testing_view',
        'Placebo_awareness',
    ]),

    ('aesthetics', [
        'Music_taste', 'Art_appreciation', 'Fashion_sense',
        'Interior_design', 'Nature_aesthetics', 'Film_taste',
        'Literature_taste', 'Architecture_sense', 'Photography_interest',
        'Dance_appreciation', 'Design_thinking', 'Color_preference',
        'Minimalism', 'Maximalism', 'Vintage_aesthetic',
        'Street_art_attitude', 'Beauty_standard', 'Typography_sense',
        'Sonic_preference', 'Texture_sensitivity', 'Symmetry_preference',
        'Light_preference', 'Handmade_value',
        'Brutalism_attitude',
    ]),

    ('education', [
        'Learning_style', 'Academic_identity', 'Knowledge_value',
        'Curiosity_trait', 'Autodidact_habit', 'Formal_education_value',
        'Mentorship_attitude', 'Exam_attitude', 'Lifelong_learning',
        'Skill_specialization', 'Generalist_identity', 'Reading_for_learning',
        'Lecture_preference', 'Hands_on_learning', 'Peer_learning',
        'Education_access_belief', 'Credential_value', 'MOOC_attitude',
        'Study_group_habit', 'Spaced_repetition',
        'Research_orientation', 'Thesis_experience', 'Teaching_identity',
        'Socratic_method',
    ]),

    ('environment', [
        'Nature_connection', 'Climate_belief', 'Sustainability_practice',
        'Recycling_habit', 'Animal_welfare', 'Conservation_value',
        'Pollution_concern', 'Carbon_awareness', 'Green_consumption',
        'Eco_anxiety', 'Outdoor_recreation', 'Rewilding_attitude',
        'Nuclear_energy_view', 'Renewable_trust', 'Water_conservation',
        'Biodiversity_awareness', 'Environmental_justice', 'Composting_habit',
        'Permaculture_interest', 'Ocean_awareness', 'Deforestation_concern',
        'Urban_greening', 'Eco_tourism',
        'Extinction_awareness',
    ]),

    # ── 4 new clusters ──

    ('relationships', [
        'Friendship_value', 'Romantic_ideal', 'Trust_building',
        'Loyalty_norm', 'Jealousy_norm', 'Breakup_coping',
        'Long_distance', 'Dating_norm', 'Intimacy_comfort',
        'Boundaries_in_love', 'Platonic_affection', 'Mentorship_bond',
        'Coworker_bond', 'Neighbor_relation', 'Online_friendship',
        'Gossip_attitude', 'Forgiveness_in_love',
        'Love_language', 'Conflict_in_love', 'Commitment_value',
        'Independence_in_rel', 'Vulnerability_openness',
        'Attachment_anxiety', 'Attachment_avoidance',
    ]),

    ('transportation', [
        'Car_culture', 'Public_transit_value', 'Cycling_commute',
        'Walking_preference', 'Flight_attitude', 'Road_rage',
        'Speed_preference', 'EV_attitude', 'Carpooling_norm',
        'Train_romance', 'Commute_tolerance', 'Traffic_patience',
        'Parking_stress', 'Navigation_app', 'Driving_pleasure',
        'Motorcycle_culture', 'Boat_affinity', 'Space_travel_dream',
        'Autonomous_vehicle', 'Ride_sharing',
        'Aviation_safety', 'Road_trip_love', 'Pedestrian_rights',
        'Hyperloop_attitude',
    ]),

    ('law_justice', [
        'Rule_of_law', 'Due_process', 'Jury_belief', 'Prison_reform',
        'Death_penalty_view', 'Police_trust', 'Vigilante_attitude',
        'Whistleblower_support', 'IP_respect', 'Privacy_right',
        'Gun_control_view', 'Drug_policy_view', 'Rehabilitation_belief',
        'Restorative_justice', 'Mandatory_minimum_view', 'Corruption_intolerance',
        'International_law', 'Human_rights_value', 'War_crimes_awareness',
        'Censorship_resistance', 'Surveillance_concern',
        'Wrongful_conviction', 'Judicial_independence', 'Civil_liberties',
    ]),

    ('creativity', [
        'Improvisation_skill', 'Brainstorming_habit', 'Creative_block',
        'Inspiration_seeking', 'Artistic_expression', 'Musical_creation',
        'Writing_creativity', 'Design_creativity', 'Problem_reframing',
        'Divergent_thinking', 'Convergent_thinking', 'Flow_state',
        'Collaboration_creativity', 'Solo_creation', 'Creative_courage',
        'Remix_culture', 'Originality_value', 'Aesthetic_intuition',
        'Cross_pollination', 'Creative_routine', 'Constraint_creativity',
        'Serendipity_openness', 'Playful_experimentation',
        'Creative_identity_self',
    ]),
])


# ============================================================
# HUB MEMES — high-degree central nodes (~20)
# ============================================================

HUB_MEMES = {
    # Original 5-cluster hubs
    'Programming', 'Science_trust', 'Rationalism', 'Meme_theory', 'Parenting',
    # Expanded/new cluster hubs
    'Native_language', 'Attachment_style', 'Core_identity', 'Social_hierarchy',
    'Democracy_value', 'Work_ethic', 'Exercise_habit', 'Music_taste',
    'Knowledge_value', 'Nature_connection', 'Friendship_value', 'Rule_of_law',
    'Divergent_thinking',
}


# ============================================================
# NEGATIVE EDGES — semantically incompatible meme pairs (~40)
# ============================================================

NEGATIVE_EDGES = [
    # Original (from 47-node graph)
    ('Rationalism', 'Traditions', -0.4),
    ('Science_trust', 'Free_will', -0.25),
    ('Career_growth', 'Home_comfort', -0.3),
    ('Meritocracy', 'Environmental_ethics', -0.2),
    ('Deadline_pressure', 'Board_games', -0.2),
    ('Stoicism', 'Music_listening', -0.15),
    ('Code_quality', 'Deadline_pressure', -0.25),
    ('Running', 'Home_comfort', -0.2),
    ('Open_source', 'Deadline_pressure', -0.15),
    ('Free_will', 'Statistical_thinking', -0.2),
    ('Rationalism', 'Marriage_vows', -0.15),
    ('Self_improvement', 'Home_comfort', -0.2),
    ('Chess', 'Photography', -0.1),
    ('Democracy', 'Meritocracy', -0.15),
    # Cross-original/new clusters
    ('Innovation_value', 'Tradition_value', -0.35),
    ('Conspiracy_openness', 'Science_trust', -0.4),
    ('Suppression_habit', 'Emotional_expression', -0.3),
    ('Optimism_bias', 'Catastrophizing', -0.35),
    ('Tech_optimism', 'Automation_fear', -0.35),
    ('Nationalism', 'Globalism', -0.4),
    ('Frugality', 'Luxury_attitude', -0.3),
    ('Alternative_medicine', 'Vaccination_trust', -0.4),
    ('Minimalism', 'Maximalism', -0.35),
    ('Formal_education_value', 'Autodidact_habit', -0.2),
    ('Democracy_value', 'Authority_deference', -0.3),
    ('Free_speech_defense', 'Censorship_attitude', -0.4),
    ('Political_cynicism', 'Activism_drive', -0.25),
    # New cluster conflicts
    ('Car_culture', 'Cycling_commute', -0.3),
    ('Speed_preference', 'Pedestrian_rights', -0.25),
    ('Prison_reform', 'Vigilante_attitude', -0.35),
    ('Death_penalty_view', 'Rehabilitation_belief', -0.4),
    ('Surveillance_concern', 'Police_trust', -0.25),
    ('Solo_creation', 'Collaboration_creativity', -0.2),
    ('Originality_value', 'Remix_culture', -0.2),
    ('Attachment_anxiety', 'Attachment_avoidance', -0.4),
    ('Independence_in_rel', 'Jealousy_norm', -0.3),
    ('Loyalty_norm', 'Gossip_attitude', -0.3),
    # Cross new-cluster conflicts
    ('Rule_of_law', 'Civil_disobedience', -0.25),
    ('Gun_control_view', 'Vigilante_attitude', -0.3),
    ('Autonomous_vehicle', 'Driving_pleasure', -0.2),
]


# ============================================================
# CROSS-CLUSTER LINKS — semantic bridges (~80)
# ============================================================

CROSS_LINKS = [
    # ── Original 19 (from 47-node build_bmc_graph) ──
    ('Meme_theory', 'Evolution_theory', 0.7),
    ('Meme_theory', 'Neuroscience_basics', 0.5),
    ('Graph_theory', 'Complexity_science', 0.6),
    ('Algorithms', 'Graph_theory', 0.6),
    ('Programming', 'Statistical_thinking', 0.4),
    ('Rationalism', 'Science_trust', 0.8),
    ('Self_improvement', 'Career_growth', 0.6),
    ('Self_improvement', 'Running', 0.4),
    ('Child_education', 'Humanism', 0.5),
    ('Reading_fiction', 'Philosophy_mind', 0.4),
    ('Chess', 'Algorithms', 0.3),
    ('Stoicism', 'Self_improvement', 0.6),
    ('Environmental_ethics', 'Parenting', 0.3),
    ('History_WW2', 'Democracy', 0.4),
    ('Cooking', 'Family_dinner', 0.5),
    ('Music_listening', 'Home_comfort', 0.3),
    ('Mentoring', 'Humanism', 0.4),
    ('Open_source', 'Meritocracy', 0.5),
    ('Technical_writing', 'Rationalism', 0.3),

    # ── Expanded originals ↔ expanded originals ──
    ('Debate_skill', 'Conflict_resolution', 0.5),
    ('Public_speaking', 'Professional_identity', 0.5),
    ('Storytelling', 'Personal_narrative', 0.6),
    ('Humor_style', 'Joy_expression', 0.4),
    ('Listening_skill', 'Empathy_skill', 0.5),
    ('Negotiation', 'Conflict_resolution', 0.6),
    ('Critical_analysis', 'Science_trust', 0.5),
    ('Time_management', 'Planning_ahead', 0.4),
    ('Financial_skill', 'Resource_management', 0.5),
    ('Resilience', 'Stress_response', 0.6),
    ('Boundary_setting', 'Personal_space', 0.5),
    ('Forgiveness_skill', 'Apology_norm', 0.4),
    ('National_identity', 'Cultural_pride', 0.6),
    ('Professional_identity', 'Career_ambition', 0.7),

    # ── New clusters ↔ original clusters ──
    ('Digital_literacy', 'Smartphone_dependence', 0.5),
    ('Privacy_norm', 'Privacy_concern', 0.6),
    ('Democracy_value', 'Fairness_norm', 0.6),
    ('Authority_respect', 'Social_hierarchy', 0.5),
    ('Work_ethic', 'Planning_ahead', 0.5),
    ('Career_ambition', 'Professional_identity', 0.7),
    ('Exercise_habit', 'Stress_response', 0.4),
    ('Body_image', 'Body_identity', 0.7),
    ('Music_taste', 'Joy_expression', 0.4),
    ('Art_appreciation', 'Creative_identity', 0.6),
    ('Knowledge_value', 'Science_trust', 0.5),
    ('Curiosity_trait', 'Metalinguistic_awareness', 0.4),
    ('Moral_framework', 'Fairness_norm', 0.7),
    ('Cooking_skill', 'Cooking', 0.8),
    ('Nature_connection', 'Nature_aesthetics', 0.6),
    ('Conservation_value', 'Moral_framework', 0.4),

    # ── New cluster ↔ new cluster ──
    ('Activism_drive', 'Environmental_justice', 0.5),
    ('Climate_belief', 'Sustainability_practice', 0.6),
    ('Voting_habit', 'Democracy_value', 0.5),
    ('Social_media_habit', 'Digital_identity', 0.7),
    ('Mindfulness_practice', 'Yoga_practice', 0.6),
    ('Charity_giving', 'Welfare_attitude', 0.5),
    ('Fitness_identity', 'Sports_identity', 0.6),
    ('Reading_fiction', 'Reading_for_learning', 0.6),
    ('Entrepreneurship', 'Innovation_value', 0.5),

    # ── Relationships cluster bridges ──
    ('Friendship_value', 'Cooperation_norm', 0.5),
    ('Romantic_ideal', 'Attachment_style', 0.7),
    ('Trust_building', 'Reciprocity', 0.6),
    ('Loyalty_norm', 'Group_identity', 0.5),
    ('Breakup_coping', 'Grief_processing', 0.5),
    ('Love_language', 'Emotional_expression', 0.5),
    ('Vulnerability_openness', 'Empathy_skill', 0.4),
    ('Attachment_anxiety', 'Attachment_style', 0.6),
    ('Mentorship_bond', 'Mentoring', 0.7),

    # ── Transportation cluster bridges ──
    ('Car_culture', 'Status_symbol', 0.4),
    ('Public_transit_value', 'Sustainability_practice', 0.5),
    ('Cycling_commute', 'Exercise_habit', 0.5),
    ('EV_attitude', 'Climate_belief', 0.5),
    ('Road_rage', 'Anger_management', 0.4),
    ('Commute_tolerance', 'Patience_skill', 0.5),

    # ── Law/justice cluster bridges ──
    ('Rule_of_law', 'Democracy_value', 0.6),
    ('Due_process', 'Fairness_norm', 0.6),
    ('Human_rights_value', 'Humanism', 0.7),
    ('Privacy_right', 'Privacy_concern', 0.6),
    ('Whistleblower_support', 'Whistleblower_norm', 0.7),
    ('Restorative_justice', 'Forgiveness_skill', 0.4),
    ('IP_respect', 'Open_source', 0.3),

    # ── Creativity cluster bridges ──
    ('Divergent_thinking', 'Problem_solving', 0.6),
    ('Creative_courage', 'Resilience', 0.4),
    ('Flow_state', 'Mindfulness_practice', 0.5),
    ('Musical_creation', 'Music_taste', 0.6),
    ('Writing_creativity', 'Writing_skill', 0.6),
    ('Cross_pollination', 'Lifelong_learning', 0.4),
    ('Playful_experimentation', 'Curiosity_trait', 0.5),
    ('Artistic_expression', 'Art_appreciation', 0.7),
    ('Design_creativity', 'Design_thinking', 0.5),
]


# ============================================================
# UTILITY → MEME CONNECTIONS (8 Panksepp systems)
# ============================================================

UTILITY_CONNECTIONS = {
    'SEEKING': [
        # Original
        ('Neuroscience_basics', 0.7, 'redirect'), ('Evolution_theory', 0.6, 'redirect'),
        ('Meme_theory', 0.8, 'redirect'), ('Graph_theory', 0.5, 'redirect'),
        ('Philosophy_mind', 0.6, 'redirect'), ('Complexity_science', 0.5, 'redirect'),
        ('Reading_fiction', 0.4, 'redirect'), ('Chess', 0.3, 'redirect'),
        ('Photography', 0.3, 'redirect'), ('Career_growth', 0.5, 'redirect'),
        # New — expanded originals
        ('Critical_analysis', 0.4, 'redirect'), ('Problem_solving', 0.5, 'redirect'),
        ('Innovation_value', 0.4, 'interpret'),
        # New — new clusters
        ('Curiosity_trait', 0.6, 'redirect'), ('Coding_interest', 0.4, 'redirect'),
        ('Knowledge_value', 0.5, 'redirect'), ('Entrepreneurship', 0.4, 'redirect'),
        ('Lifelong_learning', 0.4, 'redirect'),
        ('Divergent_thinking', 0.5, 'redirect'), ('Inspiration_seeking', 0.4, 'redirect'),
        ('Space_travel_dream', 0.3, 'redirect'), ('Systems_thinking', 0.4, 'redirect'),
    ],
    'FEAR': [
        # Original
        ('Deadline_pressure', 0.6, 'redirect'), ('Career_growth', 0.4, 'suppress'),
        ('Free_will', 0.3, 'suppress'), ('Open_source', 0.2, 'suppress'),
        ('Traditions', 0.4, 'redirect'), ('Home_comfort', 0.3, 'redirect'),
        # New
        ('Catastrophizing', 0.5, 'redirect'), ('Cybersecurity_awareness', 0.3, 'redirect'),
        ('Automation_fear', 0.5, 'redirect'), ('Eco_anxiety', 0.4, 'redirect'),
        ('Conspiracy_openness', 0.3, 'interpret'), ('Vaccination_trust', 0.3, 'interpret'),
        ('Privacy_concern', 0.4, 'redirect'),
        ('Surveillance_concern', 0.4, 'redirect'), ('Attachment_anxiety', 0.4, 'redirect'),
        ('Aviation_safety', 0.3, 'redirect'),
    ],
    'RAGE': [
        # Original
        ('Career_growth', 0.7, 'redirect'), ('Meritocracy', 0.6, 'interpret'),
        ('Code_quality', 0.5, 'redirect'), ('Technical_writing', 0.4, 'redirect'),
        ('Open_source', 0.5, 'redirect'), ('Mentoring', 0.4, 'redirect'),
        ('Democracy', 0.3, 'interpret'),
        # New
        ('Protest_attitude', 0.5, 'redirect'), ('Activism_drive', 0.4, 'redirect'),
        ('Political_cynicism', 0.3, 'interpret'), ('Conflict_resolution', 0.5, 'suppress'),
        ('Boundary_setting', 0.4, 'redirect'), ('Environmental_justice', 0.3, 'redirect'),
        ('Road_rage', 0.5, 'redirect'), ('Vigilante_attitude', 0.4, 'redirect'),
        ('Corruption_intolerance', 0.4, 'interpret'),
    ],
    'LUST': [
        # Original
        ('Marriage_vows', 0.6, 'redirect'), ('Music_listening', 0.4, 'redirect'),
        ('Photography', 0.3, 'redirect'), ('Cooking', 0.3, 'redirect'),
        ('Running', 0.3, 'redirect'),
        # New
        ('Body_identity', 0.4, 'redirect'), ('Beauty_standard', 0.3, 'redirect'),
        ('Fashion_sense', 0.3, 'redirect'), ('Dance_appreciation', 0.3, 'redirect'),
        ('Romantic_ideal', 0.5, 'redirect'), ('Intimacy_comfort', 0.5, 'redirect'),
        ('Love_language', 0.4, 'redirect'),
    ],
    'CARE': [
        # Original
        ('Parenting', 0.7, 'redirect'), ('Family_dinner', 0.6, 'redirect'),
        ('Mentoring', 0.5, 'redirect'), ('Marriage_vows', 0.5, 'redirect'),
        ('Child_education', 0.6, 'redirect'), ('Elderly_care', 0.5, 'redirect'),
        ('Humanism', 0.4, 'interpret'),
        # New
        ('Parent_identity', 0.6, 'redirect'), ('Charity_giving', 0.5, 'redirect'),
        ('Animal_welfare', 0.4, 'redirect'), ('Forgiveness_skill', 0.4, 'redirect'),
        ('Hospitality_norm', 0.4, 'redirect'),
        ('Friendship_value', 0.5, 'redirect'), ('Loyalty_norm', 0.4, 'redirect'),
        ('Human_rights_value', 0.4, 'redirect'), ('Restorative_justice', 0.3, 'redirect'),
    ],
    'PANIC_GRIEF': [
        # Original
        ('Family_dinner', 0.4, 'redirect'), ('Marriage_vows', 0.5, 'redirect'),
        ('Home_comfort', 0.4, 'redirect'), ('Traditions', 0.5, 'redirect'),
        ('Elderly_care', 0.4, 'redirect'), ('Career_growth', 0.3, 'suppress'),
        ('Self_improvement', 0.3, 'suppress'),
        # New
        ('Grief_processing', 0.7, 'redirect'), ('Attachment_style', 0.5, 'redirect'),
        ('Resilience', 0.4, 'redirect'), ('Self_soothing', 0.5, 'redirect'),
        ('Breakup_coping', 0.5, 'redirect'), ('Loneliness_management', 0.4, 'redirect'),
        ('Childhood_memory', 0.3, 'redirect'),
    ],
    'PLAY': [
        # Original
        ('Board_games', 0.6, 'redirect'), ('Chess', 0.5, 'redirect'),
        ('Music_listening', 0.5, 'redirect'), ('Cooking', 0.4, 'redirect'),
        ('Reading_fiction', 0.5, 'redirect'), ('Running', 0.4, 'redirect'),
        ('Photography', 0.4, 'redirect'), ('Stoicism', 0.3, 'suppress'),
        # New
        ('Gaming_culture', 0.5, 'redirect'), ('Sports_identity', 0.5, 'redirect'),
        ('Humor_style', 0.5, 'redirect'), ('Humor_coping', 0.4, 'redirect'),
        ('Dance_appreciation', 0.4, 'redirect'), ('Meme_culture', 0.3, 'redirect'),
        ('Outdoor_recreation', 0.4, 'redirect'),
        ('Playful_experimentation', 0.5, 'redirect'), ('Video_gaming', 0.5, 'redirect'),
        ('Martial_arts', 0.4, 'redirect'),
    ],
    'DISGUST': [
        # Original
        ('Traditions', 0.3, 'interpret'), ('Rationalism', 0.4, 'interpret'),
        ('Science_trust', 0.4, 'interpret'), ('Humanism', 0.3, 'interpret'),
        ('Environmental_ethics', 0.3, 'interpret'),
        # New
        ('Hygiene_standard', 0.5, 'interpret'), ('Corruption_tolerance', 0.4, 'interpret'),
        ('Moral_framework', 0.4, 'interpret'), ('Taboos', 0.5, 'interpret'),
        ('Fairness_norm', 0.3, 'interpret'),
        ('Corruption_intolerance', 0.4, 'interpret'), ('Vigilante_attitude', 0.3, 'interpret'),
        ('Gossip_attitude', 0.3, 'interpret'),
    ],
}


# ============================================================
# INCOMPATIBILITY SPEC (utility × meme conflicts)
# ============================================================

INCOMPAT_SPEC = {
    'SEEKING': {
        'Deadline_pressure': 0.5, 'Home_comfort': 0.4, 'Traditions': 0.3,
        # New
        'Tradition_value': 0.3, 'Suppression_habit': 0.3, 'Authority_deference': 0.2,
        'Commute_tolerance': 0.2,
    },
    'FEAR': {
        'Self_improvement': 0.4, 'Open_source': 0.3, 'Career_growth': 0.3,
        # New
        'VR_openness': 0.3, 'Public_speaking': 0.3, 'Entrepreneurship': 0.2,
        'Creative_courage': 0.3, 'Vulnerability_openness': 0.3,
        'Space_travel_dream': 0.2,
    },
    'RAGE': {
        'Humanism': 0.3, 'Stoicism': 0.5, 'Environmental_ethics': 0.2,
        # New
        'Conflict_resolution': 0.6, 'Forgiveness_skill': 0.5, 'Mindfulness_practice': 0.4,
        'Patience_skill': 0.4, 'Restorative_justice': 0.3, 'Apology_norm': 0.4,
    },
    'LUST': {
        'Stoicism': 0.4, 'Code_quality': 0.2, 'Deadline_pressure': 0.3,
        # New
        'Suppression_habit': 0.4, 'Boundary_setting': 0.3,
        'Attachment_avoidance': 0.4, 'Independence_in_rel': 0.2,
    },
    'CARE': {
        'Career_growth': 0.3, 'Deadline_pressure': 0.4, 'Meritocracy': 0.2,
        # New
        'Political_cynicism': 0.2, 'Corruption_tolerance': 0.3,
        'Vigilante_attitude': 0.2, 'Gossip_attitude': 0.2,
    },
    'PANIC_GRIEF': {
        'Self_improvement': 0.4, 'Career_growth': 0.3, 'Running': 0.2,
        # New
        'Time_management': 0.2, 'Work_ethic': 0.2, 'Optimism_bias': 0.3,
        'Startup_culture': 0.2, 'Side_hustle': 0.2,
    },
    'PLAY': {
        'Deadline_pressure': 0.6, 'Code_quality': 0.3,
        'Elderly_care': 0.3, 'Stoicism': 0.4,
        # New
        'Authority_deference': 0.3, 'Formal_education_value': 0.2,
        'Work_ethic': 0.2, 'Rule_of_law': 0.2,
    },
    'DISGUST': {
        'Meme_theory': 0.2,
        # New
        'Corruption_tolerance': 0.5, 'Conspiracy_openness': 0.3, 'Alternative_medicine': 0.3,
        'Gossip_attitude': 0.3, 'Vigilante_attitude': 0.2,
    },
}
