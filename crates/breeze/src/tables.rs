use crate::app::{Project, Task};
use crate::cli::OutputMode;
use crate::format::{render_projects as render_projects_impl, render_tasks as render_tasks_impl};

pub fn projects_render(
  projects: &[Project],
  columns: &[String],
  mode: OutputMode,
  headers: bool,
) -> Result<String, String> {
  render_projects_impl(projects, columns, mode, headers)
}

pub fn tasks_render(
  tasks: &[Task],
  columns: &[String],
  mode: OutputMode,
  headers: bool,
) -> Result<String, String> {
  render_tasks_impl(tasks, columns, mode, headers)
}
