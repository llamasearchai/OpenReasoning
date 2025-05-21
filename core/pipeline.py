"""
Core pipeline functionality for OpenReasoning.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Step(ABC):
    """Base class for pipeline steps."""

    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the step."""
        pass

    def get_name(self) -> str:
        """Get the name of the step."""
        return self.__class__.__name__


class Pipeline:
    """A pipeline of processing steps."""

    def __init__(self, steps: List[Step] = None, max_workers: int = 4):
        """Initialize a pipeline with steps."""
        self.steps = steps or []
        self.hooks = {
            "before_step": [],
            "after_step": [],
            "on_error": [],
            "on_pipeline_start": [],
            "on_pipeline_end": [],
        }
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def add_step(self, step: Step) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self

    def add_hook(self, event: str, hook: Callable) -> "Pipeline":
        """Add a hook to the pipeline."""
        if event not in self.hooks:
            raise ValueError(f"Unknown hook event: {event}")
        self.hooks[event].append(hook)
        return self

    async def run(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the pipeline with an initial context."""
        context = initial_context or {}
        context["pipeline_id"] = str(uuid.uuid4())
        context["steps"] = []
        context["start_time"] = time.time()

        # Run pipeline start hooks
        for hook in self.hooks["on_pipeline_start"]:
            context = await hook(context)

        for i, step in enumerate(self.steps):
            step_name = step.get_name()
            context["current_step"] = step_name
            context["step_index"] = i
            context["step_start_time"] = time.time()

            # Run before hooks
            for hook in self.hooks["before_step"]:
                context = await hook(context, step)

            try:
                # Run the step
                logger.debug(f"Running step {i+1}/{len(self.steps)}: {step_name}")
                context = await step.run(context)

                # Record the step result
                step_duration = time.time() - context["step_start_time"]
                context["steps"].append(
                    {
                        "name": step_name,
                        "status": "success",
                        "index": i,
                        "duration_seconds": step_duration,
                    }
                )

                # Run after hooks
                for hook in self.hooks["after_step"]:
                    context = await hook(context, step)

            except Exception as e:
                # Record the error
                step_duration = time.time() - context["step_start_time"]
                context["steps"].append(
                    {
                        "name": step_name,
                        "status": "error",
                        "index": i,
                        "error": str(e),
                        "duration_seconds": step_duration,
                    }
                )

                # Run error hooks
                for hook in self.hooks["on_error"]:
                    context = await hook(context, step, e)

                # Re-raise the exception
                logger.error(f"Error in step {step_name}: {str(e)}")
                raise

        # Run pipeline end hooks
        context["end_time"] = time.time()
        context["duration_seconds"] = context["end_time"] - context["start_time"]
        for hook in self.hooks["on_pipeline_end"]:
            context = await hook(context)

        # Clean up the context
        del context["current_step"]
        del context["step_index"]
        del context["step_start_time"]

        return context


class ParallelPipeline(Pipeline):
    """A pipeline that runs steps in parallel where possible."""

    def __init__(self, steps: List[Step] = None, max_workers: int = 4):
        """Initialize a parallel pipeline with steps."""
        super().__init__(steps, max_workers)
        self.dependency_graph = {}  # step_index -> [dependent_step_indices]

    def add_step(self, step: Step, depends_on: List[int] = None) -> "ParallelPipeline":
        """Add a step to the pipeline with optional dependencies."""
        super().add_step(step)
        step_index = len(self.steps) - 1
        self.dependency_graph[step_index] = depends_on or []
        return self

    async def run(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the pipeline with steps in parallel where possible."""
        context = initial_context or {}
        context["pipeline_id"] = str(uuid.uuid4())
        context["steps"] = []
        context["start_time"] = time.time()

        # Run pipeline start hooks
        for hook in self.hooks["on_pipeline_start"]:
            context = await hook(context)

        # Build the reverse dependency graph (which steps are blocking which)
        blocking = {i: set() for i in range(len(self.steps))}
        for step_idx, deps in self.dependency_graph.items():
            for dep in deps:
                blocking[dep].add(step_idx)

        # Track completed steps and results
        completed = set()
        results = {}

        # Run until all steps are completed
        while len(completed) < len(self.steps):
            # Find steps that can be run (all dependencies satisfied)
            runnable = []
            for i in range(len(self.steps)):
                if i not in completed and all(
                    dep in completed for dep in self.dependency_graph.get(i, [])
                ):
                    runnable.append(i)

            if not runnable:
                # This shouldn't happen if the dependency graph is valid
                raise ValueError("Circular dependency detected in pipeline steps")

            # Run steps in parallel
            tasks = []
            for step_idx in runnable:
                step = self.steps[step_idx]
                # Create a copy of the context with current results
                step_context = {**context, **results}
                step_context["current_step"] = step.get_name()
                step_context["step_index"] = step_idx
                step_context["step_start_time"] = time.time()

                # Run before hooks
                for hook in self.hooks["before_step"]:
                    step_context = await hook(step_context, step)

                # Create a task for the step
                tasks.append(self._run_step(step_idx, step, step_context))

            # Wait for all tasks to complete
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for step_idx, result in zip(runnable, step_results):
                step = self.steps[step_idx]
                step_name = step.get_name()

                if isinstance(result, Exception):
                    # Handle exception
                    context["steps"].append(
                        {
                            "name": step_name,
                            "status": "error",
                            "index": step_idx,
                            "error": str(result),
                        }
                    )

                    # Run error hooks
                    for hook in self.hooks["on_error"]:
                        context = await hook(context, step, result)

                    # Re-raise the exception
                    logger.error(f"Error in step {step_name}: {str(result)}")
                    raise result
                else:
                    # Handle success
                    step_context, step_result = result
                    results.update(step_result)

                    # Record the step result
                    step_duration = time.time() - step_context["step_start_time"]
                    context["steps"].append(
                        {
                            "name": step_name,
                            "status": "success",
                            "index": step_idx,
                            "duration_seconds": step_duration,
                        }
                    )

                    # Run after hooks
                    for hook in self.hooks["after_step"]:
                        context = await hook(context, step)

                    # Mark step as completed
                    completed.add(step_idx)

        # Merge all results into the context
        context.update(results)

        # Run pipeline end hooks
        context["end_time"] = time.time()
        context["duration_seconds"] = context["end_time"] - context["start_time"]
        for hook in self.hooks["on_pipeline_end"]:
            context = await hook(context)

        # Clean up the context
        if "current_step" in context:
            del context["current_step"]
        if "step_index" in context:
            del context["step_index"]
        if "step_start_time" in context:
            del context["step_start_time"]

        return context

    async def _run_step(
        self, step_idx: int, step: Step, context: Dict[str, Any]
    ) -> tuple:
        """Run a single step and return the updated context and step results."""
        try:
            result = await step.run(context)
            return context, result
        except Exception as e:
            logger.error(f"Error running step {step.get_name()}: {str(e)}")
            raise e
