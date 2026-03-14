# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build          # Build debug
cargo build --release # Build release
cargo run            # Run the application
cargo test           # Run tests
```

## Project Overview

This is a creative coding project using the **Nannou** framework (Rust creative-coding toolkit built on Bevy). It simulates a spring-mass system with nodes that:
- Repel each other using a force field
- Are connected by springs that pull them together
- Have internal states that evolve via a Gaussian "growth" function (similar to Lenia-style cellular automata)
- Pass state information along spring connections (message passing)

## Architecture

Single-file application (`src/main.rs`) following Nannou's app pattern:
- `model()` - Initializes window and creates initial node/spring configuration
- `update()` - Per-frame physics simulation: repulsion, spring forces, message passing, position updates
- `view()` - Renders nodes as colored circles and connections as lines

### Key Structures

- `Node` - A particle with position, velocity, bounds, and an internal "state" value (0-1) that affects its color
- `Model` - Application state containing all nodes, spring connections, and interaction state

### Physics System

- **Repulsion**: All nodes repel each other within a radius using `attract()` (with negative strength)
- **Springs**: Connected nodes are pulled together via spring forces
- **State evolution**: Each node's state is updated based on sum of connected neighbors' states through a Gaussian growth function

## Controls

- **Click and drag** - Move nodes
- **R** - Reset simulation
- **S** - Save screenshot
