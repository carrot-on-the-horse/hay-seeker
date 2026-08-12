//! Decides when `hay` may ask the operator a question.
//!
//! Searching a repository that has never been indexed is a normal first run,
//! and a person at a terminal should be offered the index instead of an error.
//! An automated caller must never be offered anything: a prompt there is a hung
//! job, not a question. So prompting requires positive evidence of a person —
//! both standard input and standard error are terminals — and any sign of a
//! continuous-integration environment withdraws it, because build agents do
//! sometimes allocate a terminal.
//!
//! Questions and answers use standard error and standard input. Standard output
//! carries the result JSON and must stay machine-readable.

use std::io::{BufRead, IsTerminal, Write};

use anyhow::{Context, Result};
use clap::ValueEnum;

/// Environment variables whose presence means an automated build.
///
/// `CI` and `CONTINUOUS_INTEGRATION` are the conventional pair; the rest are
/// the vendor variables that CI systems set even when they hand the job a
/// terminal.
const CI_VARIABLES: &[&str] = &[
    "CI",
    "CONTINUOUS_INTEGRATION",
    "BUILDKITE",
    "CIRCLECI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TF_BUILD",
];

/// Values that mean a `CI`-style variable is switched off rather than set.
const DISABLED_VALUES: &[&str] = &["", "0", "false", "no", "off"];

/// How many times an unrecognized answer is re-asked before giving up.
const ANSWER_ATTEMPTS: usize = 3;

/// What a command may do when the index it needs does not exist.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum AutoIndex {
    /// Offer to index when a person is present; fail closed otherwise.
    #[default]
    Ask,
    /// Index without asking. The setting an automated caller opts in with.
    Always,
    /// Never index implicitly; fail closed and name the command to run.
    Never,
}

/// Whether this process may put a question to a person.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Interaction {
    /// A person is at a terminal and can answer.
    Interactive,
    /// No one is watching: pipes, a daemon, or a CI job.
    Automated,
}

impl Interaction {
    /// Detects the mode from this process's terminals and environment.
    #[must_use]
    pub fn detect() -> Self {
        Self::decide(
            std::io::stdin().is_terminal(),
            std::io::stderr().is_terminal(),
            CI_VARIABLES
                .iter()
                .any(|name| std::env::var(name).is_ok_and(|value| enabled(&value))),
        )
    }

    /// The rule itself, separated from the process it reads.
    const fn decide(
        stdin_terminal: bool,
        stderr_terminal: bool,
        continuous_integration: bool,
    ) -> Self {
        if stdin_terminal && stderr_terminal && !continuous_integration {
            Self::Interactive
        } else {
            Self::Automated
        }
    }
}

/// Returns whether a `CI`-style variable's value means "on".
fn enabled(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    !DISABLED_VALUES.contains(&value.as_str())
}

/// Asks `question` on standard error and reads a yes/no answer.
///
/// Enter accepts the default of yes, matching the prompt's `[Y/n]`. End of
/// input, or [`ANSWER_ATTEMPTS`] unrecognized answers, is a no: a caller that
/// cannot answer must not be treated as having agreed. Only call this after
/// [`Interaction::detect`] reports [`Interaction::Interactive`].
///
/// # Errors
///
/// Returns an error when standard error cannot be written or standard input
/// cannot be read.
pub fn confirm(question: &str) -> Result<bool> {
    let mut input = std::io::stdin().lock();
    let mut answer = String::new();
    for _ in 0..ANSWER_ATTEMPTS {
        let mut error_output = std::io::stderr().lock();
        write!(error_output, "hay: {question} [Y/n] ").context("write prompt")?;
        error_output.flush().context("flush prompt")?;
        drop(error_output);
        answer.clear();
        let read = input.read_line(&mut answer).context("read answer")?;
        if read == 0 {
            return Ok(false);
        }
        match answer.trim().to_ascii_lowercase().as_str() {
            "" | "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => {}
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::{AutoIndex, Interaction, enabled};

    #[test]
    fn only_a_person_at_both_terminals_is_asked() {
        assert_eq!(
            Interaction::decide(true, true, false),
            Interaction::Interactive
        );
        assert_eq!(
            Interaction::decide(false, true, false),
            Interaction::Automated,
            "piped input cannot answer a question"
        );
        assert_eq!(
            Interaction::decide(true, false, false),
            Interaction::Automated,
            "a captured stderr never shows the question"
        );
    }

    /// A CI job that allocates a terminal must still never be prompted.
    #[test]
    fn continuous_integration_is_never_interactive() {
        assert_eq!(
            Interaction::decide(true, true, true),
            Interaction::Automated
        );
    }

    #[test]
    fn a_disabled_ci_variable_does_not_count_as_ci() {
        assert!(enabled("true"));
        assert!(enabled("1"));
        assert!(!enabled("false"));
        assert!(!enabled("0"));
        assert!(!enabled(" "));
    }

    #[test]
    fn asking_is_the_default_policy() {
        assert_eq!(AutoIndex::default(), AutoIndex::Ask);
    }
}
