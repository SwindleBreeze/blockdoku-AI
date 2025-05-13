# Blockdoku AI Agent

A genetic algorithm-based agent for playing Blockdoku, implemented as part of a college project.

## Project Overview

This project implements a genetic algorithm to train an agent that plays Blockdoku, a puzzle game combining elements of Tetris and Sudoku. The AI agent learns to evaluate game states and make optimal moves using a set of weighted heuristics that are evolved over generations.

## Features

- **Genetic Algorithm Implementation**: Evolves weights for game state evaluation heuristics
- **Playable Game**: Human-playable version of Blockdoku built with Pygame
- **AI Agent Testing**: Functionality to test and validate the trained agent
- **Adaptive Parameters**: Mutation and crossover rates that dynamically adjust
- **Niching**: Population diversity preservation through fitness sharing
- **Parallel Processing**: Multi-process fitness evaluation for faster training
- **Checkpointing**: Save and resume training sessions
- **Results Analysis**: Jupyter notebook for analyzing training results

## Project Structure

```
├── blockdoku_env.py           # Environment wrapper for the game
├── genetic_alg.py             # Core genetic algorithm implementation
├── AI_suggestion_improved.py  # Improved genetic algorithm implementation
├── settings.py                # AI and training settings
├── utils.py                   # Utility functions
├── test_base_weights.py       # Testing script for trained weights
├── results.ipynb              # Jupyter notebook for result analysis
├── play_human_xp.spec         # Spec file for creating executable
├── game/                      # Game implementation directory
│   ├── game_state.py          # Game state management
│   ├── grid.py                # Game grid implementation
│   ├── piece.py               # Game pieces implementation
│   └── settings.py            # Game settings
├── compiled_game/             # Compiled executable
│   └── play_human_xp.exe      # Human-playable version
├── ga_plots/                  # Generated plots from training
│   ├── ga_fitness_plot.png    # Fitness progression plot
│   ├── ga_mutation_params_plot.png  # Mutation parameters plot
│   └── ga_weights_plot.png    # Weights evolution plot
└── ga_trained_models/         # Saved models and training logs
```

## Features of the Genetic Algorithm

- **Weight-Based Evaluation**: Uses weights for heuristics like aggregate height, number of holes, bumpiness
- **Tournament Selection**: Competition-based parent selection
- **Adaptive Mutation**: Mutation rates adjust based on population diversity
- **Elitism**: Preserves best individuals across generations
- **Parallel Fitness Evaluation**: Speeds up training using multiple processes
- **Fitness Caching**: Avoids re-evaluating identical individuals

## Heuristics Used

The agent evaluates game states using multiple heuristics:
1. Aggregate Height: Total height of all columns
2. Number of Holes: Empty cells with filled cells above them
3. Bumpiness: Difference between adjacent column heights
4. Cleared Lines/Columns: Score for completing lines/columns
5. Cleared Squares: Score for completing 3x3 squares
6. Almost Full Regions: Score for nearly completed regions

## How to Run

### Requirements
- Python 3.x
- Pygame
- NumPy
- tqdm

### Training the Agent
```bash
python genetic_alg.py
```

### Advanced Training Options
```bash
python AI_suggestion_improved.py --pop-size 150 --generations 300 --parallel --adaptive-params
```

### Testing the Agent
```bash
python test_base_weights.py
```

### Playing as Human
Run the compiled executable in play_human_xp.exe or build from source.

## Results

Training results and performance analysis can be viewed in the results.ipynb notebook. The genetic algorithm shows consistent improvement in agent performance over generations, with final models achieving significantly higher scores than random play.


## Developers
- **Edin Ćehić** - ec6746@student.uni-lj.si - 63210054
- **Aljaž Justin** - aj3744@student.uni-lj.si - 63210133
