"""ClearML maintenance Click commands."""

# ClearML cleanup is an operational boundary: broad SDK failures are reported
# and the command continues cleaning the remaining resources.

from __future__ import annotations

from typing import Any

import click
from clearml import Task
from clearml.backend_api.services import projects as projects_svc, tasks as tasks_svc

from interface.cli.utils import HELP_CONTEXT, echo_json
from master_config import (
    CLEARML_API_ACCESS_KEY,
    CLEARML_API_HOST,
    CLEARML_API_SECRET_KEY,
    CLEARML_FILES_HOST,
    CLEARML_PROJECT_NAME,
    CLEARML_SERVING_ENV_FILE,
    CLEARML_SERVING_PROJECT,
    CLEARML_SERVING_TASK_FILE,
    CLEARML_WEB_HOST,
)


def clean_clearml(force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Delete ClearML projects and tasks under the configured project.

    Args:
        force: Stop active tasks before deleting.
        dry_run: Only print planned deletes.

    Returns:
        Cleanup summary.
    """
    Task.set_credentials(
        api_host=CLEARML_API_HOST,
        web_host=CLEARML_WEB_HOST,
        files_host=CLEARML_FILES_HOST,
        key=CLEARML_API_ACCESS_KEY or None,
        secret=CLEARML_API_SECRET_KEY or None,
        store_conf_file=False,
    )
    session = Task._get_default_session()

    project_name = CLEARML_PROJECT_NAME
    res = session.send(
        projects_svc.GetAllRequest(
            name=f"^{project_name.replace(' ', '.').replace('/', '.')}.*",
            only_fields=["id", "name"],
            search_hidden=True,
            _allow_extra_fields_=True,
        )
    )
    found_projects = (
        res.response.projects if res and res.response and res.response.projects else []
    )

    serving_project = CLEARML_SERVING_PROJECT
    res_serving = session.send(
        projects_svc.GetAllRequest(
            name=f"^{serving_project.replace(' ', '.').replace('/', '.')}.*",
            only_fields=["id", "name"],
            search_hidden=True,
            _allow_extra_fields_=True,
        )
    )
    serving_projects = (
        res_serving.response.projects
        if res_serving and res_serving.response and res_serving.response.projects
        else []
    )
    found_projects.extend(serving_projects)

    seen_ids: set[str] = set()
    unique_projects = []
    for project in found_projects:
        if project.id not in seen_ids:
            seen_ids.add(project.id)
            unique_projects.append(project)
    found_projects = unique_projects

    deleted_tasks: list[str] = []
    stopped_tasks: list[str] = []
    deleted_projects: list[str] = []

    for project in found_projects:
        task_res = session.send(
            tasks_svc.GetAllRequest(
                project=[project.id],
                only_fields=["id", "name", "status"],
                search_hidden=True,
                _allow_extra_fields_=True,
            )
        )
        project_tasks = (
            task_res.response.tasks
            if task_res and task_res.response and task_res.response.tasks
            else []
        )

        for task in project_tasks:
            task_name = getattr(task, "name", task.id)
            task_status = getattr(task, "status", "unknown")

            if dry_run:
                print(
                    f"[DRY-RUN] Would delete task: {task_name} "
                    f"(id={task.id}, status={task_status})"
                )
                deleted_tasks.append(task.id)
                continue

            if force and task_status in ("in_progress", "queued", "created"):
                try:
                    session.send(tasks_svc.StopRequest(task=task.id, force=True))
                    stopped_tasks.append(task.id)
                    print(f"Stopped task: {task_name} (id={task.id})")
                except Exception as exc:
                    print(f"Warning: could not stop task {task.id}: {exc}")

            try:
                session.send(
                    tasks_svc.DeleteRequest(
                        task=task.id,
                        force=True,
                        _allow_extra_fields_=True,
                    )
                )
                deleted_tasks.append(task.id)
                print(f"Deleted task: {task_name} (id={task.id})")
            except Exception as exc:
                print(f"Warning: could not delete task {task.id}: {exc}")

    for project in sorted(
        found_projects, key=lambda item: len(item.name), reverse=True
    ):
        if dry_run:
            print(f"[DRY-RUN] Would delete project: {project.name} (id={project.id})")
            deleted_projects.append(project.id)
            continue
        try:
            session.send(
                projects_svc.DeleteRequest(
                    project=project.id,
                    force=True,
                    delete_contents=True,
                    _allow_extra_fields_=True,
                )
            )
            deleted_projects.append(project.id)
            print(f"Deleted project: {project.name} (id={project.id})")
        except Exception as exc:
            print(f"Warning: could not delete project {project.id}: {exc}")

    serving_task_file = CLEARML_SERVING_TASK_FILE
    serving_env_file = CLEARML_SERVING_ENV_FILE
    for local_file in (serving_task_file, serving_env_file):
        if local_file.exists():
            if dry_run:
                print(f"[DRY-RUN] Would remove local file: {local_file}")
            else:
                local_file.unlink()
                print(f"Removed local file: {local_file}")

    action = "Would delete" if dry_run else "Deleted"
    print(f"\n{action} {len(deleted_tasks)} tasks, {len(deleted_projects)} projects.")
    if stopped_tasks:
        print(f"Force-stopped {len(stopped_tasks)} active tasks.")

    return {
        "dry_run": dry_run,
        "stopped_tasks": len(stopped_tasks),
        "deleted_tasks": len(deleted_tasks),
        "deleted_projects": len(deleted_projects),
    }


@click.command(
    name="clean-clearml",
    context_settings=HELP_CONTEXT,
    help="Delete all ClearML projects and tasks for a fresh start.",
)
@click.option(
    "--force", is_flag=True, default=False, help="Stop active tasks before deleting."
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List what would be deleted without deleting.",
)
def clean_clearml_command(force: bool, dry_run: bool) -> None:
    """Run the ClearML cleanup command.

    Args:
        force: Stop active tasks before deleting.
        dry_run: Only print planned deletes.
    """
    echo_json(clean_clearml(force=force, dry_run=dry_run))
