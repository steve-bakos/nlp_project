import os
import logging


from collections import defaultdict


class DefaultResultStore:
    """
    Base class for logging information passed under the form of a dictionary
    like `wandb.log`
    """

    def __init__(self, name=None, logger=None):
        self.logger = logger or logging.getLogger(name)

    def log(self, infos: dict):
        self.logger.info(infos)


class WandbResultStore(DefaultResultStore):
    """
    wrapper for `wandb.log`
    """

    def log(self, infos: dict):
        super().log(infos)
        if "wandb" not in locals():
            import wandb

        wandb.log(infos)


class DictResultStore(DefaultResultStore):
    """
    Logger that stores information in a dictionary
    to be retrieved at the end of a run
    """

    def __init__(self, name=None, logger=None, single_value=True):
        super(DictResultStore, self).__init__(name, logger)
        self.single_value = single_value

        if single_value:
            self._result = {}
        else:
            self._result = defaultdict(lambda: [])

    def log(self, infos: dict):
        super().log(infos)
        if self.single_value:
            self._result.update(infos)
        else:
            for k, v in infos.items():
                self._result[k].append(v)

    def get_results(self):
        if self.single_value:
            return self._result
        else:
            return {k: v[0] if len(v) == 1 else v for k, v in self._result.items()}


def post_on_slack(message: str, thread_ts=None):
    webhook = os.getenv("SLACK_WEBHOOK")
    channel = os.getenv("SLACK_CHANNEL")
    slack_oauth = os.getenv("SLACK_OAUTH")

    if webhook is None:
        return

    if "requests" not in locals():
        import requests

    if channel is not None and slack_oauth is not None:
        res = requests.post(
            "https://slack.com/api/chat.postMessage",
            json={
                "channel": channel,
                "text": message,
                **({"thread_ts": thread_ts} if thread_ts is not None else {}),
            },
            headers={"Authorization": f"Bearer {slack_oauth}"},
        )
        try:
            result = res.json().get("ts")
        except:
            logging.error(f"slack alert: could not retrieve `ts`: {res.content}")
            result = None
        return result

    requests.post(webhook, json={"text": message})

    return None
